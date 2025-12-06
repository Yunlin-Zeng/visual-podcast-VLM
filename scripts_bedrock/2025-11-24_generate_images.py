#!/usr/bin/env python3
"""
Generate images from extracted metadata using Stable Diffusion 3.5 on Bedrock
- Reads metadata files from extract_excerpts_2025-11-24.py
- Generates images using SD3.5
- Resume capability (skips existing images)
- Rate limiting with exponential backoff

Usage:
    python 2025-11-24_generate_images.py --metadata-dir excerpts_metadata_2025-11-24 --workers 2
"""

import requests
import json
import os
import re
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path
import argparse

# Bedrock configuration - always read from file (avoid stale env var)
token_file = Path(__file__).parent / '.bedrock_token'
if token_file.exists():
    TOKEN = token_file.read_text().strip()
else:
    raise Exception(".bedrock_token not found")

SD_URL = "https://bedrock-runtime.us-west-2.amazonaws.com/model/stability.sd3-5-large-v1:0/invoke"


def clean_prompt(prompt):
    """Clean up prompt - remove redundant Scene X: prefix"""
    # Remove "Scene X: " prefix if present (handles "Scene 1: Scene 1: ..." duplication)
    cleaned = re.sub(r'^Scene \d+:\s*', '', prompt)
    # Remove again in case of duplication
    cleaned = re.sub(r'^Scene \d+:\s*', '', cleaned)
    return cleaned.strip()


def generate_image(prompt, max_retries=1):  # Changed from 3 to 1 to skip retries
    """Generate image via Stable Diffusion 3.5 with retries"""
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    # Clean and format prompt
    clean = clean_prompt(prompt)
    full_prompt = f"Photorealistic photograph, high quality, realistic lighting, no cartoon, no illustration: {clean}"

    body = {
        "prompt": full_prompt,
        "mode": "text-to-image",
        "aspect_ratio": "16:9",
        "output_format": "png"
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(SD_URL, headers=headers, json=body, timeout=120)

            if response.status_code == 200:
                result = response.json()
                return base64.b64decode(result['images'][0])
            elif response.status_code == 429:  # Rate limit
                wait_time = 2 ** (attempt + 1)
                print(f"    Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                # Print full error for debugging
                print(f"    SD API error: {response.status_code} - {response.text[:300]}")
                # Commented out retry delay - fail fast
                # if attempt < max_retries - 1:
                #     time.sleep(2)
                continue

        except Exception as e:
            print(f"    SD exception (attempt {attempt+1}): {e}")
            # Commented out retry delay - fail fast
            # time.sleep(2)

    return None


def process_scene(task, output_dir):
    """Generate one image from scene metadata"""
    episode_idx, excerpt_idx, scene_data = task

    scene_idx = scene_data['scene_idx']
    prompt = scene_data['prompt']
    filename = scene_data['filename']
    output_path = output_dir / filename

    # Skip if already generated
    if output_path.exists():
        return {
            'episode_idx': episode_idx,
            'excerpt_idx': excerpt_idx,
            'scene_idx': scene_idx,
            'status': 'skipped',
            'filename': filename
        }

    print(f"  [{episode_idx}:{excerpt_idx}:{scene_idx}] Generating: {clean_prompt(prompt)[:60]}...")

    # Generate image
    image_data = generate_image(prompt)

    if image_data:
        with open(output_path, 'wb') as f:
            f.write(image_data)
        print(f"  [{episode_idx}:{excerpt_idx}:{scene_idx}] ✓ Saved ({len(image_data)//1024}KB)")
        return {
            'episode_idx': episode_idx,
            'excerpt_idx': excerpt_idx,
            'scene_idx': scene_idx,
            'status': 'success',
            'filename': filename,
            'size': len(image_data)
        }
    else:
        print(f"  [{episode_idx}:{excerpt_idx}:{scene_idx}] ✗ Failed")
        return {
            'episode_idx': episode_idx,
            'excerpt_idx': excerpt_idx,
            'scene_idx': scene_idx,
            'status': 'failed',
            'filename': filename
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-dir', type=str, default='excerpts_metadata_2025-11-24',
                        help='Directory containing metadata JSON files')
    parser.add_argument('--output-dir', type=str, default='dataset_output_2025-11-24',
                        help='Directory to save generated images')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of parallel workers (keep low to avoid rate limiting)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between image generations (seconds)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to generate (None = all)')
    args = parser.parse_args()

    # Setup directories
    base_dir = Path(__file__).parent
    metadata_dir = base_dir / args.metadata_dir
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Generate Images from Metadata (2025-11-24)")
    print("=" * 80)
    print(f"Metadata dir: {metadata_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Parallel workers: {args.workers}")
    print(f"Delay between images: {args.delay}s")
    if args.max_images:
        print(f"Max images: {args.max_images}")
    print("=" * 80)

    # Load all metadata files
    metadata_files = sorted(metadata_dir.glob("ep*_metadata.json"))
    print(f"\nFound {len(metadata_files)} metadata files")

    if len(metadata_files) == 0:
        print("No metadata files found. Run extract_excerpts_2025-11-24.py first.")
        return

    # Collect all scenes to generate
    all_tasks = []
    for metadata_file in metadata_files:
        with open(metadata_file) as f:
            metadata = json.load(f)

        episode_idx = metadata['episode_idx']
        for excerpt in metadata['excerpts']:
            excerpt_idx = excerpt['excerpt_idx']
            for scene in excerpt['scenes']:
                all_tasks.append((episode_idx, excerpt_idx, scene))

                if args.max_images and len(all_tasks) >= args.max_images:
                    break
            if args.max_images and len(all_tasks) >= args.max_images:
                break
        if args.max_images and len(all_tasks) >= args.max_images:
            break

    print(f"Total scenes to process: {len(all_tasks)}")

    # Count already generated
    existing = sum(1 for task in all_tasks if (output_dir / task[2]['filename']).exists())
    remaining = len(all_tasks) - existing
    print(f"Already generated: {existing}")
    print(f"Remaining: {remaining}\n")

    if remaining == 0:
        print("All images already generated!")
        return

    # Process with limited parallelism
    results = []
    start_time = time.time()
    generated_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for task in all_tasks:
            future = executor.submit(process_scene, task, output_dir)
            futures.append(future)
            time.sleep(args.delay)

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result['status'] == 'success':
                generated_count += 1

            # Print progress every 10 images
            completed = len(results)
            if completed % 10 == 0 or completed == len(all_tasks):
                elapsed = time.time() - start_time
                rate = generated_count / elapsed * 60 if elapsed > 0 else 0
                print(f"\n--- Progress: {completed}/{len(all_tasks)} processed, {generated_count} generated ({rate:.1f} img/min) ---")

    # Summary
    elapsed = time.time() - start_time
    success = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    skipped = [r for r in results if r['status'] == 'skipped']

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total processed: {len(results)}")
    print(f"Generated: {len(success)} images in {elapsed/60:.1f} minutes")
    print(f"Failed: {len(failed)} images")
    print(f"Skipped (already exist): {len(skipped)} images")

    if len(success) > 0:
        total_size = sum(r.get('size', 0) for r in success)
        print(f"\nTotal size: {total_size / 1024 / 1024:.1f} MB")
        print(f"Avg size: {total_size / len(success) / 1024:.1f} KB per image")

    # Save summary
    summary = {
        'total': len(results),
        'success': len(success),
        'failed': len(failed),
        'skipped': len(skipped),
        'elapsed_seconds': elapsed,
        'failed_files': [r['filename'] for r in failed]
    }

    summary_path = output_dir / 'generation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Images saved to: {output_dir}/")

    if len(failed) > 0:
        print(f"\n⚠️  {len(failed)} images failed. Re-run script to retry.")


if __name__ == '__main__':
    main()
