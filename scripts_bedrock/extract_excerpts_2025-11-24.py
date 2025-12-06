#!/usr/bin/env python3
"""
Extract visual excerpts from podcasts with improved quality filtering
- Stricter quality requirements (no repetition, no inappropriate content)
- Ideally 700-850 words (can be shorter but not longer)
- Filters for Stable Diffusion 3.5 compatibility
- Target: 3000 qualified excerpts

Usage:
    python extract_excerpts_2025-11-24.py --target-excerpts 3000 --workers 10
"""

import requests
import json
import os
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
import time
from pathlib import Path
import argparse

# Bedrock configuration - always read from file
token_file = Path(__file__).parent / '.bedrock_token'
if token_file.exists():
    TOKEN = token_file.read_text().strip()
else:
    raise Exception(".bedrock_token not found")

CLAUDE_URL = "https://bedrock-runtime.us-west-2.amazonaws.com/model/global.anthropic.claude-sonnet-4-5-20250929-v1:0/invoke"

# Output directory
OUTPUT_DIR = Path("/home/ubuntu/image-to-text/bedrock_api_pipeline/excerpts_metadata_2025-11-24")
OUTPUT_DIR.mkdir(exist_ok=True)

EXTRACTION_PROMPT = """You are analyzing a podcast transcript to extract ALL visually-rich conversational excerpts.

The transcript is formatted with speaker labels like "Speaker 1:", "Speaker 2:", etc. with double line breaks between turns.

Your task:
1. Find ALL excerpts that are EASY TO VISUALIZE (ideally 700-850 words)
2. For each excerpt, generate 5 specific scene descriptions for image generation

Requirements for each excerpt:
- IDEALLY 700-850 words (can be slightly shorter if needed to avoid cutting mid-speech, and please try your best not to exceed 850 words)
- Contains concrete visual details (objects, places, actions, people, colors)
- Natural conversational flow with speaker turns preserved
- IMPORTANT: Keep the speaker labels (Speaker 1:, Speaker 2:) and double line breaks (\\n\\n) between turns
- Stays in one location/event (don't jump between unrelated scenes)
- NOT abstract discussions (avoid philosophy, concepts, theories)
- Do NOT cut in the middle of a speaker's speech - complete the sentence

Quality filters - SKIP excerpts that have:
- Any phrase repeated more than 3 times (indicates transcript errors or filler loops)
- Excessive filler words throughout (um, uh, like, you know used excessively)
- Markers like [inaudible], [sigh], [laughter], [music], or similar annotations appearing more than twice
- Empty or near-empty speaker turns (e.g., "Speaker 2: ." or "Speaker 1: -")
- Speakers talking over each other with fragmented/incomplete sentences throughout
- Corrupted or garbled text that doesn't form coherent sentences

Content safety filters - SKIP excerpts that contain:
- Sexual or suggestive content (nudity, underwear scenes, intimate situations)
- Graphic violence or gore
- Hate speech, slurs, or discriminatory language
- Drug use or illegal activities described in detail
- Any content that would be inappropriate for image generation with Stable Diffusion 3.5
- Profanity used excessively (occasional mild language is acceptable)

Coherence requirements:
- Conversation must have a clear topic or narrative thread
- Speakers should be having a genuine exchange (not monologue disguised as dialogue)
- The excerpt should tell a mini-story or describe a specific experience/event

Requirements for 5 scene descriptions per excerpt:
- Each scene should be a specific visual moment from the excerpt
- Describe what would be visible in a photo
- Include concrete details: objects, people, settings, actions
- Scenes should flow chronologically through the excerpt
- Scene descriptions must be safe for image generation (no inappropriate imagery)

If the transcript has NO visualizable excerpts that meet ALL quality requirements, respond with:
{{"visualizable": false, "reason": "explanation"}}

If it has visualizable excerpts, respond with:
{{
  "visualizable": true,
  "excerpts": [
    {{
      "excerpt": "700-850 word excerpt here with Speaker labels preserved",
      "word_count": 750,
      "scenes": [
        "Scene 1: detailed visual description",
        "Scene 2: detailed visual description",
        "Scene 3: detailed visual description",
        "Scene 4: detailed visual description",
        "Scene 5: detailed visual description"
      ]
    }},
    {{
      "excerpt": "another excerpt if found",
      "word_count": 780,
      "scenes": [...]
    }}
  ]
}}

Transcript to analyze:
{transcript}
"""


def call_claude(prompt, max_retries=3, debug_first_call=[True]):
    """Call Claude via Bedrock API with retries"""
    if debug_first_call[0]:
        print(f"  [DEBUG] Token length: {len(TOKEN)}, first 40 chars: {TOKEN[:40]}...")
        debug_first_call[0] = False

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8192,
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
            elif response.status_code == 429:
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


def validate_excerpt(excerpt_text, word_count):
    """Additional validation after Claude extraction"""
    # Check word count (allow shorter, but not longer than 900)
    if word_count > 900:
        return False, "too_long"

    if word_count < 400:
        return False, "too_short"

    # Check for repetition patterns (3-word phrases repeated >5 times)
    words = excerpt_text.split()
    if len(words) >= 50:
        trigrams = [' '.join(words[j:j+3]) for j in range(len(words)-2)]
        trigram_counts = Counter(trigrams)
        max_rep = max(trigram_counts.values())
        if max_rep > 5:
            return False, f"repetition_{max_rep}x"

    # Check for problematic markers
    problematic = ['[inaudible]', '[sigh]', '[music]', '[laughter]']
    marker_count = sum(excerpt_text.lower().count(m) for m in problematic)
    if marker_count > 2:
        return False, "too_many_markers"

    return True, "valid"


def process_podcast(episode_data, stats):
    """Process one podcast episode - extract visual excerpts only"""
    episode_idx, episode_url, turns = episode_data

    try:
        # Build formatted transcript from speaker turns
        transcript_parts = []
        speaker_map = {}
        speaker_counter = 1

        for turn in turns:
            speaker_id = turn.get('speaker', ['UNKNOWN'])[0]
            text = turn.get('turnText', '').strip()

            if text.startswith('[Music]') or len(text) < 10:
                continue

            if speaker_id not in speaker_map:
                speaker_map[speaker_id] = f"Speaker {speaker_counter}"
                speaker_counter += 1

            speaker_label = speaker_map[speaker_id]
            transcript_parts.append(f"{speaker_label}: {text}")

        transcript = "\n\n".join(transcript_parts)

        if len(transcript) < 2000:
            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'skipped',
                'reason': 'too_short'
            }

        # Skip very long episodes (>50k chars) - they have 40% higher failure rate
        if len(transcript) > 50000:
            print(f"\n[{episode_idx}] Skipping episode ({len(transcript)} chars) - too long")
            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'skipped',
                'reason': 'too_long'
            }

        print(f"\n[{episode_idx}] Processing episode ({len(transcript)} chars, {len(speaker_map)} speakers)")

        prompt = EXTRACTION_PROMPT.format(transcript=transcript)
        response = call_claude(prompt)

        if not response:
            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'failed',
                'reason': 'claude_api_error'
            }

        response_text = response.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"[{episode_idx}] JSON parse error: {e}")
            # Log failed response for debugging
            failed_log_path = OUTPUT_DIR / 'failed_responses'
            os.makedirs(failed_log_path, exist_ok=True)
            with open(failed_log_path / f'ep{episode_idx}_failed.txt', 'w') as f:
                f.write(f"Episode: {episode_idx}\n")
                f.write(f"Error: {e}\n")
                f.write(f"Response length: {len(response_text)}\n")
                f.write(f"Response preview: {response_text[:500] if response_text else '(empty)'}\n")
            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'failed',
                'reason': 'json_parse_error'
            }

        if not result.get('visualizable', False):
            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'not_visualizable',
                'reason': result.get('reason', 'unknown')
            }

        excerpts = result.get('excerpts', [])
        if not excerpts:
            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'not_visualizable',
                'reason': 'no_excerpts'
            }

        # Process and validate excerpts
        excerpts_data = []
        for excerpt_idx, excerpt_item in enumerate(excerpts):
            excerpt_text = excerpt_item.get('excerpt', '')
            scenes = excerpt_item.get('scenes', [])
            word_count = excerpt_item.get('word_count', len(excerpt_text.split()))

            if not excerpt_text or not scenes or len(scenes) < 5:
                continue

            # Validate excerpt
            is_valid, reason = validate_excerpt(excerpt_text, word_count)
            if not is_valid:
                print(f"[{episode_idx}] Excerpt {excerpt_idx} rejected: {reason}")
                stats['rejected_excerpts'] += 1
                continue

            scene_data = []
            for scene_idx, scene_prompt in enumerate(scenes[:5], 1):
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
                'word_count': word_count,
                'scenes': scene_data
            })

        if excerpts_data:
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

            num_excerpts = len(excerpts_data)
            stats['total_excerpts'] += num_excerpts
            print(f"[{episode_idx}] Saved {num_excerpts} excerpts (Total: {stats['total_excerpts']})")

            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'success',
                'num_excerpts': num_excerpts,
                'metadata_file': str(metadata_path)
            }
        else:
            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'failed',
                'reason': 'no_valid_excerpts'
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
    parser = argparse.ArgumentParser(description="Extract visual excerpts from podcasts (2025-11-24)")
    parser.add_argument('--target-excerpts', type=int, default=3000, help='Target number of excerpts')
    parser.add_argument('--start-episode', type=int, default=0, help='Starting episode index')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers')
    parser.add_argument('--batch-size', type=int, default=100, help='Episodes per batch')
    parser.add_argument('--dataset', default='/home/ubuntu/image-to-text/data/SPoRC/speakerTurnData.jsonl.gz')
    parser.add_argument('--max-episodes', type=int, default=15000,
                        help='Max episodes to load')
    args = parser.parse_args()

    # Count existing excerpts from previous runs
    existing_files = list(OUTPUT_DIR.glob("ep*_metadata.json"))
    existing_excerpts = 0
    processed_episodes = set()
    for f in existing_files:
        try:
            with open(f) as fp:
                meta = json.load(fp)
                existing_excerpts += meta.get('num_excerpts', 0)
                processed_episodes.add(meta.get('episode_idx'))
        except:
            pass

    print("=" * 80)
    print("Extract Visual Excerpts (2025-11-24 - Improved Quality)")
    print("=" * 80)
    print(f"Target excerpts: {args.target_excerpts}")
    print(f"Existing excerpts from previous runs: {existing_excerpts}")
    print(f"Already processed episodes: {len(processed_episodes)}")
    print(f"Remaining to extract: {max(0, args.target_excerpts - existing_excerpts)}")
    print(f"Starting episode: {args.start_episode}")
    print(f"Parallel workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)

    # Check if already done
    if existing_excerpts >= args.target_excerpts:
        print(f"\nAlready have {existing_excerpts} excerpts (target: {args.target_excerpts}). Done!")
        return

    # Load episodes from dataset (up to max_episodes)
    print(f"\nLoading up to {args.max_episodes} episodes from {args.dataset}...")
    episodes_by_url = defaultdict(list)

    with gzip.open(args.dataset, 'rt') as f:
        for i, line in enumerate(f):
            try:
                turn = json.loads(line)
                url = turn.get('mp3url', '')
                text = turn.get('turnText', '').strip()

                if url and len(text) > 10:
                    episodes_by_url[url].append(turn)

                if len(episodes_by_url) >= args.max_episodes:
                    print(f"  Reached {args.max_episodes} episodes, stopping load...")
                    break

                if i % 500000 == 0 and i > 0:
                    print(f"  Processed {i} turns, found {len(episodes_by_url)} episodes...")
            except:
                continue

    # Filter episodes with enough turns
    all_episodes = []
    for url, turns in episodes_by_url.items():
        if len(turns) >= 5:
            all_episodes.append((url, turns))

    all_episodes.sort(key=lambda x: x[0])
    print(f"Found {len(all_episodes)} total episodes with speaker turns")

    # Process in batches until we reach target (counting existing excerpts)
    stats = {'total_excerpts': existing_excerpts, 'rejected_excerpts': 0}
    all_results = []
    current_episode = args.start_episode
    start_time = time.time()

    while stats['total_excerpts'] < args.target_excerpts and current_episode < len(all_episodes):
        batch_end = min(current_episode + args.batch_size, len(all_episodes))
        batch_episodes = []

        for idx in range(current_episode, batch_end):
            # Skip already processed episodes
            if idx in processed_episodes:
                continue
            url, turns = all_episodes[idx]
            batch_episodes.append((idx, url, turns))

        print(f"\n{'='*80}")
        print(f"Processing batch: episodes {current_episode} to {batch_end-1}")
        print(f"Current total excerpts: {stats['total_excerpts']} / {args.target_excerpts}")
        print(f"{'='*80}")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_podcast, ep, stats): ep for ep in batch_episodes}

            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)

                if stats['total_excerpts'] >= args.target_excerpts:
                    print(f"\n Target of {args.target_excerpts} excerpts reached!")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

        current_episode = batch_end

        # Save progress
        progress_summary = {
            'total_excerpts': stats['total_excerpts'],
            'rejected_excerpts': stats['rejected_excerpts'],
            'episodes_processed': len(all_results),
            'current_episode_idx': current_episode,
            'target': args.target_excerpts
        }
        with open(OUTPUT_DIR / 'progress.json', 'w') as f:
            json.dump(progress_summary, f, indent=2)

    # Final summary
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    success = [r for r in all_results if r['status'] == 'success']
    not_visual = [r for r in all_results if r['status'] == 'not_visualizable']
    failed = [r for r in all_results if r['status'] == 'failed']
    skipped = [r for r in all_results if r['status'] == 'skipped']

    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Episodes processed: {len(all_results)}")
    print(f"  - Success: {len(success)}")
    print(f"  - Not visualizable: {len(not_visual)}")
    print(f"  - Failed: {len(failed)}")
    print(f"  - Skipped: {len(skipped)}")
    print(f"\nTotal qualified excerpts: {stats['total_excerpts']}")
    print(f"Rejected excerpts: {stats['rejected_excerpts']}")
    print(f"Total scenes to generate: {stats['total_excerpts'] * 5}")

    if len(success) > 0:
        print(f"\nVisual rate: {len(success) / len(all_results) * 100:.1f}%")
        print(f"Avg excerpts per visual episode: {stats['total_excerpts'] / len(success):.2f}")

    # Save final summary
    summary = {
        'target_excerpts': args.target_excerpts,
        'total_excerpts': stats['total_excerpts'],
        'rejected_excerpts': stats['rejected_excerpts'],
        'episodes_processed': len(all_results),
        'success': len(success),
        'not_visualizable': len(not_visual),
        'failed': len(failed),
        'skipped': len(skipped),
        'elapsed_seconds': elapsed,
        'results': all_results
    }

    summary_path = OUTPUT_DIR / 'extraction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Metadata files saved to: {OUTPUT_DIR}/")
    print("\nNext step: Run generate_images.py to create images from metadata")


if __name__ == '__main__':
    main()
