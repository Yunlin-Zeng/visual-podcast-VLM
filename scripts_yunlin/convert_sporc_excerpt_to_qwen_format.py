#!/usr/bin/env python3
"""
Convert SPoRC podcast excerpt dataset to Qwen3-VL training format.
Supports multiple data sources (old and new extractions).
Includes ALL excerpts with at least 1 image (variable number of images per sample).
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Default data sources: (metadata_dir, image_dir, name)
DEFAULT_DATA_SOURCES = [
    (
        "/home/ubuntu/image-to-text/bedrock_api_pipeline/2025-11-03_excerpts_metadata",
        "/home/ubuntu/image-to-text/bedrock_api_pipeline/2025-11-03_image_generation",
        "2025-11-03"
    ),
    (
        "/home/ubuntu/image-to-text/bedrock_api_pipeline/2025-11-24_excerpts_metadata",
        "/home/ubuntu/image-to-text/bedrock_api_pipeline/2025-11-25_image_generation",
        "2025-11-24"
    ),
]

OUTPUT_FILE = Path("/home/ubuntu/image-to-text/Qwen3-VL/data/qwen_training_data_sporc_4004_samples.json")


def load_all_metadata(metadata_dir: Path, source_name: str) -> List[Dict[str, Any]]:
    """Load all metadata JSON files from a directory."""
    all_excerpts = []

    metadata_files = sorted(metadata_dir.glob("ep*_metadata.json"))
    print(f"  [{source_name}] Found {len(metadata_files)} metadata files")

    for meta_file in metadata_files:
        with open(meta_file, 'r') as f:
            data = json.load(f)
            episode_idx = data['episode_idx']

            for excerpt in data['excerpts']:
                excerpt_data = {
                    'episode_idx': episode_idx,
                    'excerpt_idx': excerpt['excerpt_idx'],
                    'excerpt_text': excerpt['excerpt'],
                    'scenes': excerpt['scenes'],
                    'source': source_name
                }
                all_excerpts.append(excerpt_data)

    print(f"  [{source_name}] Loaded {len(all_excerpts)} total excerpts")
    return all_excerpts


def get_existing_images(excerpt: Dict[str, Any], image_dir: Path) -> List[str]:
    """Get all existing images for an excerpt. Returns list of image paths."""
    image_paths = []

    for scene in excerpt['scenes']:
        filename = scene['filename']
        image_path = image_dir / filename

        if image_path.exists():
            # Use absolute path for reliability
            image_paths.append(str(image_path))

    return image_paths


def convert_to_qwen_format(
    excerpts: List[Dict[str, Any]],
    image_dir: Path,
    source_name: str,
    prompt_template: str
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Convert excerpts to Qwen3-VL training format. Includes all excerpts with at least 1 image."""
    qwen_data = []
    # Track by number of images: {1: count, 2: count, ..., 6: count, 0: count}
    stats = {i: 0 for i in range(7)}

    for excerpt in excerpts:
        image_paths = get_existing_images(excerpt, image_dir)
        num_images = len(image_paths)

        # Skip excerpts with no images
        if num_images == 0:
            stats[0] += 1
            continue

        stats[num_images] += 1

        # Build the conversation format with variable number of image tags
        image_tags = "<image>" * num_images

        qwen_entry = {
            "image": image_paths,
            "conversations": [
                {
                    "from": "human",
                    "value": f"{image_tags}\n{prompt_template}"
                },
                {
                    "from": "gpt",
                    "value": excerpt['excerpt_text']
                }
            ]
        }
        qwen_data.append(qwen_entry)

    # Print stats
    print(f"  [{source_name}] By image count:")
    for i in range(6, 0, -1):
        if stats[i] > 0:
            print(f"    {i} images: {stats[i]}")
    if stats[0] > 0:
        print(f"    0 images (skipped): {stats[0]}")

    return qwen_data, stats


DEFAULT_PROMPT = "Generate a natural conversational podcast dialogue. Use the format Speaker 1:, Speaker 2:, Speaker 3:, etc. for multiple speakers. Do not reference the images or use phrases like \"our first image\". Write casual, authentic spoken dialogue without introductions or sign-offs. The word count should be around 800 words."


def main():
    parser = argparse.ArgumentParser(description='Convert SPoRC excerpts to Qwen training format')
    parser.add_argument('--output', type=str, default=str(OUTPUT_FILE), help='Output JSON file')
    parser.add_argument('--sources', type=str, nargs='*', help='Data sources in format: metadata_dir:image_dir:name')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, help='Prompt template for training')
    parser.add_argument('--prompt-file', type=str, help='Read prompt from file (overrides --prompt)')
    args = parser.parse_args()

    output_file = Path(args.output)

    # Get prompt template
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt_template = f.read().strip()
        print(f"Loaded prompt from: {args.prompt_file}")
    else:
        prompt_template = args.prompt

    # Parse data sources
    if args.sources:
        data_sources = []
        for src in args.sources:
            parts = src.split(':')
            if len(parts) != 3:
                print(f"Invalid source format: {src}")
                print("Expected format: metadata_dir:image_dir:name")
                return
            data_sources.append((parts[0], parts[1], parts[2]))
    else:
        data_sources = DEFAULT_DATA_SOURCES

    print("=" * 80)
    print("Converting SPoRC excerpts to Qwen3-VL format (ALL excerpts with >=1 image)")
    print("=" * 80)
    print(f"Output file: {output_file}")
    print(f"Data sources: {len(data_sources)}")
    print(f"Prompt: {prompt_template[:80]}...")
    print("-" * 80)

    # Process all data sources
    all_qwen_data = []
    total_stats = {i: 0 for i in range(7)}

    for metadata_dir, image_dir, name in data_sources:
        metadata_path = Path(metadata_dir)
        image_path = Path(image_dir)

        if not metadata_path.exists():
            print(f"  [{name}] Warning: Metadata directory not found: {metadata_path}")
            continue
        if not image_path.exists():
            print(f"  [{name}] Warning: Image directory not found: {image_path}")
            continue

        print(f"\nProcessing: {name}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Images: {image_path}")

        # Load and convert
        excerpts = load_all_metadata(metadata_path, name)
        qwen_data, stats = convert_to_qwen_format(excerpts, image_path, name, prompt_template)

        all_qwen_data.extend(qwen_data)
        for i in range(7):
            total_stats[i] += stats[i]

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_included = sum(total_stats[i] for i in range(1, 7))
    print(f"Total excerpts included: {total_included}")
    print("By image count:")
    for i in range(6, 0, -1):
        if total_stats[i] > 0:
            print(f"  {i} images: {total_stats[i]}")
    if total_stats[0] > 0:
        print(f"  0 images (skipped): {total_stats[0]}")

    # Save to output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_qwen_data, f, indent=2)

    print(f"\nSaved {len(all_qwen_data)} training examples to {output_file}")

    # Show a sample entry
    if all_qwen_data:
        print("\n" + "-" * 80)
        print("Sample training entry:")
        print("-" * 80)
        sample = all_qwen_data[0]
        print(f"Number of images: {len(sample['image'])}")
        print(f"Image paths: {sample['image'][0]}... (showing first)")
        print(f"GPT response length: {len(sample['conversations'][1]['value'])} chars")


if __name__ == "__main__":
    main()
