#!/usr/bin/env python3
"""
Visual Podcast Evaluation Script
Computes metrics comparing 32B Finetuned vs 235B Base model outputs.

Metrics:
1. CLIPScore - Visual grounding (average across 5 images per sample)
2. Distinct-2 - Bi-gram lexical diversity
3. Podcast Style Metrics - Turn length and speaker switch rate

Usage:
    python evaluate_metrics.py --data_dir ./data
    python evaluate_metrics.py --data_dir ./data --skip_clip  # Skip slow CLIPScore
"""

import argparse
import os
import re
import string
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")


def load_transcript(filepath: Path) -> str:
    """Load transcript from file, extracting only the dialogue content."""
    if not filepath.exists():
        return ""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip header lines (Sample:, Word count:, Generation time:, ===)
    lines = content.split('\n')
    dialogue_lines = []
    in_dialogue = False

    for line in lines:
        # Skip header information
        if line.startswith('Sample:') or line.startswith('Word count:') or \
           line.startswith('Generation time:') or line.startswith('==='):
            in_dialogue = True
            continue
        if in_dialogue:
            dialogue_lines.append(line)

    return '\n'.join(dialogue_lines).strip() if dialogue_lines else content.strip()


def normalize_text(text: str) -> str:
    """Normalize text: lowercase and remove punctuation."""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def compute_distinct_n(text: str, n: int = 2) -> float:
    """
    Compute Distinct-N score (bi-gram diversity by default).
    Formula: Count of Unique N-grams / Total Count of N-grams
    """
    # Normalize text
    normalized = normalize_text(text)

    # Tokenize on whitespace
    tokens = normalized.split()

    if len(tokens) < n:
        return 0.0

    # Generate n-grams
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    if len(ngrams) == 0:
        return 0.0

    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)

    return unique_ngrams / total_ngrams


def parse_speaker_turns(text: str) -> list:
    """
    Parse speaker turns from transcript.
    Looks for patterns like "Speaker 1:", "Host:", "Alex:", "[Name]:", etc.
    Returns list of (speaker, text) tuples.
    """
    # Pattern to match speaker labels at start of line
    # Matches: "Speaker 1:", "Host:", "Alex:", "[Name]:", etc.
    speaker_pattern = r'^(?:\[?)(Speaker\s*\d+|Host|Guest|Co-host|[A-Z][a-z]+)(?:\]?):\s*'

    lines = text.split('\n')
    turns = []
    current_speaker = None
    current_text = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if this line starts a new speaker turn
        match = re.match(speaker_pattern, line)
        if match:
            # Save previous turn
            if current_speaker and current_text:
                turns.append((current_speaker, ' '.join(current_text)))

            current_speaker = match.group(1)
            remaining_text = line[match.end():].strip()
            current_text = [remaining_text] if remaining_text else []
        else:
            # Continue current speaker's turn
            if current_speaker:
                current_text.append(line)

    # Save last turn
    if current_speaker and current_text:
        turns.append((current_speaker, ' '.join(current_text)))

    return turns


def compute_podcast_stats(text: str) -> dict:
    """
    Compute podcast-style statistics.
    Returns:
        - avg_turn_length: Average words per speaker turn
        - switch_rate: Speaker switches per 1,000 words
        - num_turns: Total number of speaker turns
    """
    turns = parse_speaker_turns(text)

    if len(turns) == 0:
        return {
            'avg_turn_length': 0.0,
            'switch_rate': 0.0,
            'num_turns': 0
        }

    # Calculate turn lengths
    turn_lengths = []
    for speaker, turn_text in turns:
        words = turn_text.split()
        turn_lengths.append(len(words))

    avg_turn_length = sum(turn_lengths) / len(turn_lengths) if turn_lengths else 0.0

    # Calculate switch rate (switches per 1,000 words)
    total_words = sum(turn_lengths)
    num_switches = len(turns) - 1  # Switches = turns - 1

    switch_rate = (num_switches / total_words * 1000) if total_words > 0 else 0.0

    return {
        'avg_turn_length': avg_turn_length,
        'switch_rate': switch_rate,
        'num_turns': len(turns)
    }


def compute_clipscore(data_dir: Path) -> tuple:
    """
    Compute CLIPScore for all samples.
    For each sample: average CLIPScore of (full transcript vs each of 5 images)
    Returns two lists: scores for 32B and 235B models.
    """
    try:
        import torch
        from PIL import Image
        from torchmetrics.multimodal.clip_score import CLIPScore
        import torchvision.transforms as T

        print("Loading CLIP model on CPU (this may take a moment)...")

        # Force CPU
        device = torch.device('cpu')
        clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")
        clip_metric = clip_metric.to(device)
        clip_metric.eval()

        # Image transform
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 255).to(torch.uint8))
        ])

        scores_32b = []
        scores_235b = []

        sample_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sample_')])

        for sample_dir in tqdm(sample_dirs, desc="Computing CLIPScore (CPU)"):
            # Load images
            images = []
            # Try different naming patterns
            image_files = sorted(sample_dir.glob("image_*.jpg")) + sorted(sample_dir.glob("image_*.png"))
            if not image_files:
                image_files = sorted(sample_dir.glob("*.jpg")) + sorted(sample_dir.glob("*.png"))

            for img_path in image_files[:5]:  # Max 5 images
                try:
                    # Follow symlinks
                    actual_path = img_path.resolve()
                    img = Image.open(actual_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"Warning: Could not load {img_path}: {e}")

            if len(images) == 0:
                print(f"Warning: No images found in {sample_dir.name}, skipping CLIPScore")
                scores_32b.append(None)
                scores_235b.append(None)
                continue

            # Load transcripts
            transcript_32b = load_transcript(sample_dir / "transcript_32b.txt")
            transcript_235b = load_transcript(sample_dir / "transcript_235b.txt")

            # Compute scores for each model
            for transcript, scores_list, model_name in [
                (transcript_32b, scores_32b, "32B"),
                (transcript_235b, scores_235b, "235B")
            ]:
                if not transcript:
                    print(f"Warning: Missing {model_name} transcript in {sample_dir.name}")
                    scores_list.append(None)
                    continue

                # Compute CLIPScore for full transcript vs each image, then average
                image_scores = []

                # Truncate text for CLIP (77 token limit, ~300 chars safe)
                text = transcript[:500]

                for img in images:
                    try:
                        img_tensor = transform(img).unsqueeze(0)

                        with torch.no_grad():
                            clip_metric.update(img_tensor, [text])
                            score = clip_metric.compute().item()
                            clip_metric.reset()

                        image_scores.append(score)
                    except Exception as e:
                        print(f"Warning: CLIPScore error for {sample_dir.name}: {e}")

                if image_scores:
                    avg_score = sum(image_scores) / len(image_scores)
                    scores_list.append(avg_score)
                else:
                    scores_list.append(None)

        return scores_32b, scores_235b

    except ImportError as e:
        print(f"\nError: CLIP dependencies not available: {e}")
        print("Install with: pip install torch torchmetrics transformers pillow torchvision")
        print("Skipping CLIPScore computation.\n")
        return [], []
    except Exception as e:
        print(f"\nError computing CLIPScore: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def main():
    parser = argparse.ArgumentParser(description="Evaluate Visual Podcast metrics")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Root directory containing sample subfolders")
    parser.add_argument("--output", type=str, default="evaluation_results.csv",
                        help="Output CSV file path")
    parser.add_argument("--skip_clip", action="store_true",
                        help="Skip CLIPScore computation (useful for quick testing)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist")
        return

    # Get all sample directories
    sample_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sample_')])
    print(f"Found {len(sample_dirs)} samples in {data_dir}")

    if len(sample_dirs) == 0:
        print("No sample directories found!")
        return

    # Initialize results storage
    results = []

    # Compute Distinct-2 and Podcast Stats for each sample
    print("\nComputing Distinct-2 and Podcast Style Metrics...")
    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        sample_name = sample_dir.name

        # Load transcripts
        transcript_32b = load_transcript(sample_dir / "transcript_32b.txt")
        transcript_235b = load_transcript(sample_dir / "transcript_235b.txt")

        # Check for missing files
        if not transcript_32b:
            print(f"Warning: Missing transcript_32b.txt in {sample_name}")
        if not transcript_235b:
            print(f"Warning: Missing transcript_235b.txt in {sample_name}")

        # Compute Distinct-2
        distinct2_32b = compute_distinct_n(transcript_32b, n=2) if transcript_32b else None
        distinct2_235b = compute_distinct_n(transcript_235b, n=2) if transcript_235b else None

        # Compute Podcast Stats
        stats_32b = compute_podcast_stats(transcript_32b) if transcript_32b else {'avg_turn_length': 0, 'switch_rate': 0, 'num_turns': 0}
        stats_235b = compute_podcast_stats(transcript_235b) if transcript_235b else {'avg_turn_length': 0, 'switch_rate': 0, 'num_turns': 0}

        # Word count
        words_32b = len(transcript_32b.split()) if transcript_32b else 0
        words_235b = len(transcript_235b.split()) if transcript_235b else 0

        results.append({
            'sample': sample_name,
            'words_32b': words_32b,
            'words_235b': words_235b,
            'distinct2_32b': distinct2_32b,
            'distinct2_235b': distinct2_235b,
            'avg_turn_length_32b': stats_32b['avg_turn_length'],
            'avg_turn_length_235b': stats_235b['avg_turn_length'],
            'switch_rate_32b': stats_32b['switch_rate'],
            'switch_rate_235b': stats_235b['switch_rate'],
            'num_turns_32b': stats_32b['num_turns'],
            'num_turns_235b': stats_235b['num_turns'],
        })

    # Compute CLIPScore (optional, slow on CPU)
    if not args.skip_clip:
        print("\nComputing CLIPScore (this may take 10-20 minutes on CPU)...")
        clip_32b, clip_235b = compute_clipscore(data_dir)

        if clip_32b and clip_235b:
            for i, result in enumerate(results):
                if i < len(clip_32b):
                    result['clipscore_32b'] = clip_32b[i]
                    result['clipscore_235b'] = clip_235b[i]
    else:
        print("\nSkipping CLIPScore computation (--skip_clip flag set)")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save per-sample results
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(f"\nPer-sample results saved to: {output_path}")

    # Compute and display summary statistics
    print("\n" + "="*80)
    print("EVALUATION SUMMARY: 32B Finetuned vs 235B Base")
    print("="*80)

    # Create summary comparison
    summary_data = {
        'Metric': [],
        '32B_Finetuned': [],
        '235B_Base': [],
        'Difference': []
    }

    # Helper function to add metric
    def add_metric(name, col_32b, col_235b, fmt=".2f"):
        vals_32b = df[col_32b].dropna()
        vals_235b = df[col_235b].dropna()

        if len(vals_32b) == 0 or len(vals_235b) == 0:
            return

        mean_32b = vals_32b.mean()
        mean_235b = vals_235b.mean()
        diff = mean_32b - mean_235b

        summary_data['Metric'].append(name)
        summary_data['32B_Finetuned'].append(f"{mean_32b:{fmt}}")
        summary_data['235B_Base'].append(f"{mean_235b:{fmt}}")
        summary_data['Difference'].append(f"{diff:+{fmt}}")

    # Add all metrics
    add_metric('Word Count', 'words_32b', 'words_235b', ".1f")
    add_metric('Distinct-2 (Lexical Diversity)', 'distinct2_32b', 'distinct2_235b', ".4f")
    add_metric('Avg Turn Length (words)', 'avg_turn_length_32b', 'avg_turn_length_235b', ".1f")
    add_metric('Switch Rate (per 1000 words)', 'switch_rate_32b', 'switch_rate_235b', ".1f")
    add_metric('Number of Turns', 'num_turns_32b', 'num_turns_235b', ".1f")

    if 'clipscore_32b' in df.columns:
        add_metric('CLIPScore (Visual Grounding)', 'clipscore_32b', 'clipscore_235b', ".2f")

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Save summary
    summary_path = output_path.parent / f"{output_path.stem}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Additional insights
    print("\n" + "-"*80)
    print("INTERPRETATION:")
    print("-"*80)

    # Compare turn lengths
    avg_turn_32b = df['avg_turn_length_32b'].mean()
    avg_turn_235b = df['avg_turn_length_235b'].mean()
    if avg_turn_32b > avg_turn_235b:
        print(f"- 32B produces longer speaker turns ({avg_turn_32b:.1f} vs {avg_turn_235b:.1f} words)")
        print("  → More substantial, flowing dialogue")
    else:
        print(f"- 235B produces longer speaker turns ({avg_turn_235b:.1f} vs {avg_turn_32b:.1f} words)")

    # Compare switch rates
    switch_32b = df['switch_rate_32b'].mean()
    switch_235b = df['switch_rate_235b'].mean()
    if switch_32b < switch_235b:
        print(f"- 32B has lower switch rate ({switch_32b:.1f} vs {switch_235b:.1f} per 1000 words)")
        print("  → More natural conversation flow, less fragmented")
    else:
        print(f"- 235B has lower switch rate ({switch_235b:.1f} vs {switch_32b:.1f} per 1000 words)")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
