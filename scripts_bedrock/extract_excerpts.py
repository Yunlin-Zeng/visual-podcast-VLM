#!/usr/bin/env python3
"""
Stage 1: Extract visual excerpts and scene prompts from podcasts (no image generation)
- Uses speaker-turn data for structured dialogue format
- Extracts ALL visualizable excerpts from each podcast
- Generates 5 scene descriptions per excerpt
- Saves metadata only (no images yet)
- Processes podcasts in parallel
"""

import requests
import json
import os
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import time
from pathlib import Path

# Bedrock configuration
TOKEN = os.environ.get('AWS_BEARER_TOKEN_BEDROCK')
if not TOKEN:
    # Try reading from file
    token_file = Path(__file__).parent / '.bedrock_token'
    if token_file.exists():
        TOKEN = token_file.read_text().strip()
    else:
        raise Exception("AWS_BEARER_TOKEN_BEDROCK not set and .bedrock_token not found")

CLAUDE_URL = "https://bedrock-runtime.us-west-2.amazonaws.com/model/global.anthropic.claude-sonnet-4-5-20250929-v1:0/invoke"

# Output directory - will be set based on today's date or command line arg
# Format: YYYY-MM-DD_excerpts_metadata
from datetime import datetime
DEFAULT_OUTPUT_DIR = Path(f"/home/ubuntu/image-to-text/bedrock_api_pipeline/{datetime.now().strftime('%Y-%m-%d')}_excerpts_metadata")
OUTPUT_DIR = None  # Set in main() based on args

EXTRACTION_PROMPT = """You are analyzing a podcast transcript to extract ALL visually-rich conversational excerpts.

The transcript is formatted with speaker labels like "Speaker 1:", "Speaker 2:", etc. with double line breaks between turns.

Your task:
1. Find ALL excerpts that are 600-800 words and EASY TO VISUALIZE (you may find multiple excerpts in one transcript)
2. For each excerpt, generate 5 specific scene descriptions for image generation

Requirements for each excerpt:
- 600-800 words (can be slightly shorter/longer if needed to avoid cutting mid-speech)
- Contains concrete visual details (objects, places, actions, people, colors)
- Natural conversational flow with speaker turns preserved
- IMPORTANT: Keep the speaker labels (Speaker 1:, Speaker 2:) and double line breaks (\\n\\n) between turns
- Stays in one location/event (don't jump between unrelated scenes)
- NOT abstract discussions (avoid philosophy, concepts, theories)
- Do NOT cut in the middle of a speaker's speech - complete the sentence

Requirements for 5 scene descriptions per excerpt:
- Each scene should be a specific visual moment from the excerpt
- Describe what would be visible in a photo
- Include concrete details: objects, people, settings, actions
- Scenes should flow chronologically through the excerpt

If the transcript has NO visualizable excerpts, respond with:
{{"visualizable": false, "reason": "explanation"}}

If it has visualizable excerpts, respond with:
{{{{
  "visualizable": true,
  "excerpts": [
    {{
      "excerpt": "600-800 word excerpt here",
      "scenes": [
        "Scene 1: detailed visual description",
        "Scene 2: detailed visual description",
        "Scene 3: detailed visual description",
        "Scene 4: detailed visual description",
        "Scene 5: detailed visual description"
      ]
    }},
    {{
      "excerpt": "another 600-800 word excerpt if found",
      "scenes": [...]
    }}
  ]
}}}}

Transcript to analyze:
{transcript}
"""


def call_claude(prompt, max_retries=3, debug_first_call=[True]):
    """Call Claude via Bedrock API with retries"""
    # Debug: Print token info on first call only
    if debug_first_call[0]:
        print(f"  [DEBUG] Token length: {len(TOKEN)}, first 40 chars: {TOKEN[:40]}")
        debug_first_call[0] = False

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8192,  # Increased for multiple excerpts
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(CLAUDE_URL, headers=headers, json=body, timeout=120)

            if response.status_code == 200:
                result = response.json()
                return result['content'][0]['text']
            elif response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  Claude API error: {response.status_code} - {response.text[:200]}")
                return None

        except Exception as e:
            print(f"  Claude exception (attempt {attempt+1}): {e}")
            time.sleep(1)

    return None


def process_podcast(episode_data):
    """Process one podcast episode - extract visual excerpts only"""
    episode_idx, episode_url, turns = episode_data

    try:
        # Build formatted transcript from speaker turns
        transcript_parts = []
        speaker_map = {}  # Map original speaker IDs to simple labels
        speaker_counter = 1

        for turn in turns:
            speaker_id = turn.get('speaker', ['UNKNOWN'])[0]
            text = turn.get('turnText', '').strip()

            # Skip music/noise
            if text.startswith('[Music]') or len(text) < 10:
                continue

            # Assign speaker label
            if speaker_id not in speaker_map:
                speaker_map[speaker_id] = f"Speaker {speaker_counter}"
                speaker_counter += 1

            speaker_label = speaker_map[speaker_id]
            transcript_parts.append(f"{speaker_label}: {text}")

        # Join with double line breaks for clear separation
        transcript = "\n\n".join(transcript_parts)

        if len(transcript) < 2000:  # Skip very short episodes
            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'skipped',
                'reason': 'too_short'
            }

        print(f"\n[{episode_idx}] Processing episode ({len(transcript)} chars, {len(speaker_map)} speakers)")

        # Extract visual excerpts using Claude
        prompt = EXTRACTION_PROMPT.format(transcript=transcript)
        response = call_claude(prompt)

        if not response:
            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'failed',
                'reason': 'claude_api_error'
            }

        # Parse response
        response_text = response.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"[{episode_idx}] JSON parse error: {e}")
            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'failed',
                'reason': 'json_parse_error'
            }

        # Check if visualizable
        if not result.get('visualizable', False):
            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'not_visualizable',
                'reason': result.get('reason', 'unknown')
            }

        # Process excerpts
        excerpts = result.get('excerpts', [])
        if not excerpts:
            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'not_visualizable',
                'reason': 'no_excerpts'
            }

        print(f"[{episode_idx}] Found {len(excerpts)} visual excerpts")

        # Save metadata
        excerpts_data = []
        for excerpt_idx, excerpt_item in enumerate(excerpts):
            excerpt_text = excerpt_item.get('excerpt', '')
            scenes = excerpt_item.get('scenes', [])

            if not excerpt_text or not scenes:
                continue

            # Create scene metadata
            scene_data = []
            for scene_idx, scene_prompt in enumerate(scenes, 1):
                scene_data.append({
                    'scene_idx': scene_idx,
                    'prompt': scene_prompt,
                    'filename': f"ep{episode_idx}_ex{excerpt_idx}_scene{scene_idx}.png",
                    'generated': False
                })

            excerpts_data.append({
                'excerpt_idx': excerpt_idx,
                'excerpt': excerpt_text,
                'excerpt_length': len(excerpt_text),
                'word_count': len(excerpt_text.split()),
                'scenes': scene_data
            })

        if excerpts_data:
            # Save metadata
            metadata = {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'original_transcript_length': len(transcript),
                'num_excerpts': len(excerpts_data),
                'excerpts': excerpts_data
            }

            metadata_path = OUTPUT_DIR / f"ep{episode_idx}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'success',
                'num_excerpts': len(excerpts_data),
                'metadata_file': str(metadata_path)
            }
        else:
            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'failed',
                'reason': 'no_complete_excerpts'
            }

    except Exception as e:
        print(f"[{episode_idx}] Exception: {e}")
        import traceback
        traceback.print_exc()
        return {
            'episode_idx': episode_idx,
            'status': 'error',
            'error': str(e)
        }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of episodes to process')
    parser.add_argument('--start-episode', type=int, default=0, help='Starting episode index')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers')
    parser.add_argument('--dataset', default='/home/ubuntu/image-to-text/data/SPoRC/speakerTurnData.jsonl.gz')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: YYYY-MM-DD_excerpts_metadata)')
    args = parser.parse_args()

    # Set output directory
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("Stage 1: Extract Visual Excerpts (Speaker-Turn Format)")
    print("=" * 80)
    print(f"Starting episode: {args.start_episode}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Parallel workers: {args.workers}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)

    # Load speaker turns and group by episode (only load what we need)
    print(f"\nLoading speaker turns from {args.dataset}...")
    episodes_by_url = defaultdict(list)
    target_episodes = (args.start_episode + args.num_episodes) * 2  # Load 2x to account for filtering

    with gzip.open(args.dataset, 'rt') as f:
        for i, line in enumerate(f):
            try:
                turn = json.loads(line)
                url = turn.get('mp3url', '')
                text = turn.get('turnText', '').strip()

                # Only collect turns with real content
                if url and len(text) > 10:
                    episodes_by_url[url].append(turn)

                # Stop when we have enough episodes
                if len(episodes_by_url) >= target_episodes:
                    print(f"  Collected {len(episodes_by_url)} episodes, stopping read...")
                    break

                # Progress indicator
                if i % 100000 == 0 and i > 0:
                    print(f"  Processed {i} turns, found {len(episodes_by_url)} episodes...")
            except:
                continue

    # Convert to list and filter by min turns
    all_episodes = []
    for url, turns in episodes_by_url.items():
        # Need at least 5 meaningful turns for a conversation
        if len(turns) >= 5:
            all_episodes.append((url, turns))

    # Sort by URL for consistency and apply start/limit
    all_episodes.sort(key=lambda x: x[0])
    episodes = []
    for idx in range(args.start_episode, min(args.start_episode + args.num_episodes, len(all_episodes))):
        url, turns = all_episodes[idx]
        episodes.append((idx, url, turns))

    print(f"Found {len(all_episodes)} total episodes with speaker turns")
    print(f"Processing episodes {args.start_episode} to {args.start_episode + len(episodes) - 1}")

    # Process in parallel
    print(f"\nProcessing with {args.workers} workers...")
    print("(Extraction only - no images)\n")

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_podcast, ep): ep for ep in episodes}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            # Print progress
            completed = len(results)
            if completed % 10 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (len(episodes) - completed) / rate if rate > 0 else 0
                print(f"\nProgress: {completed}/{len(episodes)} episodes ({rate:.1f} eps/min, ~{remaining/60:.1f}min remaining)")

    # Summary
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("SUMMARY - EXTRACTION STAGE")
    print("=" * 80)

    success = [r for r in results if r['status'] == 'success']
    not_visual = [r for r in results if r['status'] == 'not_visualizable']
    failed = [r for r in results if r['status'] == 'failed']
    skipped = [r for r in results if r['status'] == 'skipped']

    total_excerpts = sum(r.get('num_excerpts', 0) for r in success)
    total_scenes = total_excerpts * 5

    print(f"Processed: {len(results)} episodes in {elapsed/60:.1f} minutes")
    print(f"Success: {len(success)} episodes → {total_excerpts} excerpts → {total_scenes} scenes to generate")
    print(f"Not visualizable: {len(not_visual)} episodes")
    print(f"Failed: {len(failed)} episodes")
    print(f"Skipped: {len(skipped)} episodes (too short)")

    if len(success) > 0:
        print(f"\nVisual rate: {len(success) / len(results) * 100:.1f}%")
        print(f"Avg excerpts per visual episode: {total_excerpts / len(success):.2f}")

    # Save summary
    summary = {
        'processed': len(results),
        'success': len(success),
        'not_visualizable': len(not_visual),
        'failed': len(failed),
        'skipped': len(skipped),
        'total_excerpts': total_excerpts,
        'total_scenes': total_scenes,
        'elapsed_seconds': elapsed,
        'results': results
    }

    summary_path = OUTPUT_DIR / 'extraction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Metadata files saved to: {OUTPUT_DIR}/")
    print("\nNext step: Run generate_images.py to create images from metadata")


if __name__ == '__main__':
    main()
