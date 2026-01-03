#!/usr/bin/env python3
"""
Parallelized pipeline to generate podcast training dataset
- Extracts ALL visualizable excerpts from each podcast
- Generates 5 images per excerpt
- Processes podcasts in parallel
"""

import requests
import json
import os
import gzip
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
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
SD_URL = "https://bedrock-runtime.us-west-2.amazonaws.com/model/stability.sd3-5-large-v1:0/invoke"

# Output directory
OUTPUT_DIR = Path("/home/ubuntu/image-to-text/bedrock_api_pipeline/dataset_output")
OUTPUT_DIR.mkdir(exist_ok=True)

EXTRACTION_PROMPT = """You are analyzing a podcast transcript to extract ALL visually-rich conversational excerpts.

Your task:
1. Find ALL excerpts that are 600-800 words and EASY TO VISUALIZE (you may find multiple excerpts in one transcript)
2. For each excerpt, generate 5 specific scene descriptions for image generation

Requirements for each excerpt:
- 600-800 words (can be slightly shorter/longer if needed to avoid cutting mid-speech)
- Contains concrete visual details (objects, places, actions, people, colors)
- Natural conversational flow with speaker turns preserved
- IMPORTANT: Use \\n\\n (double line breaks) to separate different speakers or natural paragraph breaks
- Format speaker turns clearly for readability
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


def generate_image(prompt, max_retries=3):
    """Generate image via Stable Diffusion with retries"""
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    body = {
        "prompt": f"Photorealistic photograph, high quality, realistic lighting, no cartoon, no illustration: {prompt}",
        "mode": "text-to-image",
        "aspect_ratio": "16:9",
        "output_format": "png"
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(SD_URL, headers=headers, json=body, timeout=60)

            if response.status_code == 200:
                result = response.json()
                return base64.b64decode(result['images'][0])
            elif response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt
                print(f"  SD rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  SD API error: {response.status_code}")
                return None

        except Exception as e:
            print(f"  SD exception (attempt {attempt+1}): {e}")
            time.sleep(1)

    return None


def process_podcast(episode_data):
    """Process one podcast episode - extract all visual excerpts and generate images"""
    episode_idx, episode = episode_data

    try:
        transcript = episode.get('transcript', '')
        episode_url = episode.get('mp3_url', episode.get('mp3url', f'episode_{episode_idx}'))

        if len(transcript) < 2000:  # Skip very short episodes
            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'skipped',
                'reason': 'too_short'
            }

        print(f"\n[{episode_idx}] Processing episode ({len(transcript)} chars)")

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
        except json.JSONDecodeError:
            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'failed',
                'reason': 'json_parse_error'
            }

        # Check if visualizable
        if not result.get('visualizable'):
            return {
                'episode_idx': episode_idx,
                'episode_url': episode_url,
                'status': 'not_visualizable',
                'reason': result.get('reason', 'unknown')
            }

        # Process each excerpt
        excerpts_data = []
        excerpt_count = len(result.get('excerpts', []))
        print(f"[{episode_idx}] Found {excerpt_count} visual excerpts")

        for excerpt_idx, excerpt_data in enumerate(result['excerpts']):
            excerpt_text = excerpt_data.get('excerpt', '')
            scenes = excerpt_data.get('scenes', [])

            if len(scenes) != 5:
                print(f"[{episode_idx}:{excerpt_idx}] Warning: {len(scenes)} scenes instead of 5")
                continue

            # Generate images for this excerpt
            print(f"[{episode_idx}:{excerpt_idx}] Generating 5 images...")
            images = []

            for scene_idx, scene_prompt in enumerate(scenes):
                image_data = generate_image(scene_prompt)

                if image_data:
                    # Save image
                    image_filename = f"ep{episode_idx}_ex{excerpt_idx}_scene{scene_idx+1}.png"
                    image_path = OUTPUT_DIR / image_filename

                    with open(image_path, 'wb') as f:
                        f.write(image_data)

                    images.append({
                        'scene_idx': scene_idx + 1,
                        'prompt': scene_prompt,
                        'filename': image_filename
                    })
                else:
                    print(f"[{episode_idx}:{excerpt_idx}] Failed to generate image {scene_idx+1}")

            if len(images) == 5:  # Only include if all images generated
                excerpts_data.append({
                    'excerpt_idx': excerpt_idx,
                    'excerpt': excerpt_text,
                    'excerpt_length': len(excerpt_text),
                    'word_count': len(excerpt_text.split()),
                    'scenes': images
                })
                print(f"[{episode_idx}:{excerpt_idx}] ✓ Complete ({len(excerpt_text)} chars)")

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
    parser.add_argument('--dataset', default='/home/ubuntu/image-to-text/data/SPoRC/episodeLevelDataSample.jsonl.gz')
    args = parser.parse_args()

    print("=" * 80)
    print("Podcast Dataset Generation Pipeline")
    print("=" * 80)
    print(f"Starting episode: {args.start_episode}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Parallel workers: {args.workers}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)

    # Load episodes
    print(f"\nLoading episodes from {args.dataset}...")
    episodes = []

    with gzip.open(args.dataset, 'rt') as f:
        for i, line in enumerate(f):
            if i < args.start_episode:
                continue
            if i >= args.start_episode + args.num_episodes:
                break
            try:
                episode = json.loads(line)
                episodes.append((i, episode))
            except:
                continue

    print(f"Loaded {len(episodes)} episodes")

    # Process in parallel
    print(f"\nProcessing with {args.workers} workers...")
    print("(This will take approximately 10-30 minutes for 100 episodes)\n")

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
    print("SUMMARY")
    print("=" * 80)

    success = [r for r in results if r['status'] == 'success']
    not_visual = [r for r in results if r['status'] == 'not_visualizable']
    failed = [r for r in results if r['status'] in ['failed', 'error']]
    skipped = [r for r in results if r['status'] == 'skipped']

    total_excerpts = sum(r.get('num_excerpts', 0) for r in success)

    print(f"Processed: {len(episodes)} episodes in {elapsed/60:.1f} minutes")
    print(f"")
    print(f"✓ Success: {len(success)} episodes → {total_excerpts} visual excerpts")
    print(f"✗ Not visualizable: {len(not_visual)} episodes")
    print(f"✗ Failed: {len(failed)} episodes")
    print(f"- Skipped: {len(skipped)} episodes")
    print(f"")
    print(f"Visual rate: {len(success)/len(episodes)*100:.1f}% of episodes")
    print(f"Excerpts per visual episode: {total_excerpts/len(success):.1f}" if success else "N/A")

    # Save summary
    summary = {
        'config': {
            'num_episodes': args.num_episodes,
            'workers': args.workers,
            'dataset': args.dataset
        },
        'stats': {
            'total_episodes': len(episodes),
            'success': len(success),
            'total_excerpts': total_excerpts,
            'not_visualizable': len(not_visual),
            'failed': len(failed),
            'skipped': len(skipped),
            'visual_rate': len(success)/len(episodes) if episodes else 0,
            'excerpts_per_episode': total_excerpts/len(success) if success else 0,
            'elapsed_minutes': elapsed/60
        },
        'results': results
    }

    summary_path = OUTPUT_DIR / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Dataset saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
