#!/usr/bin/env python3
"""
Stage 2: Generate images from extracted metadata
- Reads metadata files created by extract_excerpts.py
- Generates images using Stable Diffusion 3.5
- Limited parallelism to avoid rate limiting
- Can resume from failures
"""

import requests
import json
import os
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

SD_URL = "https://bedrock-runtime.us-west-2.amazonaws.com/model/stability.sd3-5-large-v1:0/invoke"

# Directories - will be set from command line args
# Format: YYYY-MM-DD_excerpts_metadata for metadata, YYYY-MM-DD_image_generation for images
from datetime import datetime
METADATA_DIR = None  # Set in main() based on args
IMAGE_OUTPUT_DIR = None  # Set in main() based on args


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
                if 'images' in result and result['images']:
                    return base64.b64decode(result['images'][0])
                else:
                    # Content filtered - no retry will help
                    reason = result.get('finish_reasons', ['unknown'])[0]
                    print(f"    Content filtered: {reason}")
                    return None
            elif response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt
                print(f"    Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"    SD API error: {response.status_code}")
                return None

        except Exception as e:
            print(f"    SD exception (attempt {attempt+1}): {e}")
            time.sleep(1)

    return None


def process_scene(task):
    """Generate one image from scene metadata"""
    episode_idx, excerpt_idx, scene_data = task

    scene_idx = scene_data['scene_idx']
    prompt = scene_data['prompt']
    filename = scene_data['filename']
    output_path = IMAGE_OUTPUT_DIR / filename

    # Skip if already generated
    if output_path.exists():
        print(f"  [{episode_idx}:{excerpt_idx}:{scene_idx}] Already exists, skipping")
        return {
            'episode_idx': episode_idx,
            'excerpt_idx': excerpt_idx,
            'scene_idx': scene_idx,
            'status': 'skipped',
            'filename': filename
        }

    print(f"  [{episode_idx}:{excerpt_idx}:{scene_idx}] Generating...")

    # Generate image
    image_data = generate_image(prompt)

    if image_data:
        with open(output_path, 'wb') as f:
            f.write(image_data)
        print(f"  [{episode_idx}:{excerpt_idx}:{scene_idx}] ✓ Saved ({len(image_data)} bytes)")
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers (keep low to avoid rate limiting)')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between image generations (seconds)')
    parser.add_argument('--max-excerpts', type=int, default=None, help='Maximum number of excerpts to generate images for (None = all)')
    parser.add_argument('--metadata-dir', type=str, required=True, help='Directory containing excerpt metadata files')
    parser.add_argument('--image-dir', type=str, default=None, help='Output directory for images (default: YYYY-MM-DD_image_generation)')
    args = parser.parse_args()

    # Set directories
    global METADATA_DIR, IMAGE_OUTPUT_DIR
    METADATA_DIR = Path(args.metadata_dir)
    if args.image_dir:
        IMAGE_OUTPUT_DIR = Path(args.image_dir)
    else:
        IMAGE_OUTPUT_DIR = Path(f"/home/ubuntu/image-to-text/bedrock_api_pipeline/{datetime.now().strftime('%Y-%m-%d')}_image_generation")
    IMAGE_OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("Stage 2: Generate Images from Metadata")
    print("=" * 80)
    print(f"Metadata dir: {METADATA_DIR}")
    print(f"Image output: {IMAGE_OUTPUT_DIR}")
    print(f"Parallel workers: {args.workers}")
    print(f"Delay between images: {args.delay}s")
    if args.max_excerpts:
        print(f"Max excerpts: {args.max_excerpts} ({args.max_excerpts * 5} images)")
    print("=" * 80)

    # Load all metadata files
    metadata_files = sorted(METADATA_DIR.glob("ep*_metadata.json"))
    print(f"\nFound {len(metadata_files)} metadata files")

    if len(metadata_files) == 0:
        print("No metadata files found. Run extract_excerpts.py first.")
        return

    # Collect all scenes to generate
    all_tasks = []
    excerpts_collected = 0

    for metadata_file in metadata_files:
        if args.max_excerpts and excerpts_collected >= args.max_excerpts:
            break

        with open(metadata_file) as f:
            metadata = json.load(f)

        episode_idx = metadata['episode_idx']
        for excerpt in metadata['excerpts']:
            if args.max_excerpts and excerpts_collected >= args.max_excerpts:
                break

            excerpt_idx = excerpt['excerpt_idx']
            for scene in excerpt['scenes']:
                all_tasks.append((episode_idx, excerpt_idx, scene))

            excerpts_collected += 1

    print(f"Total excerpts: {excerpts_collected}")
    print(f"Total scenes to generate: {len(all_tasks)}")

    # Count already generated
    existing = sum(1 for task in all_tasks if (IMAGE_OUTPUT_DIR / task[2]['filename']).exists())
    print(f"Already generated: {existing}")
    print(f"Remaining: {len(all_tasks) - existing}\n")

    # Process with limited parallelism
    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for task in all_tasks:
            # Submit task
            future = executor.submit(process_scene, task)
            futures.append(future)

            # Add delay to avoid rate limiting
            time.sleep(args.delay)

        # Collect results
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)

            # Print progress
            completed = len(results)
            if completed % 50 == 0 or completed == len(all_tasks):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (len(all_tasks) - completed) / rate if rate > 0 else 0
                print(f"\nProgress: {completed}/{len(all_tasks)} images ({rate:.1f} img/min, ~{remaining/60:.1f}min remaining)")

    # Summary
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("SUMMARY - IMAGE GENERATION")
    print("=" * 80)

    success = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    skipped = [r for r in results if r['status'] == 'skipped']

    print(f"Total images: {len(results)}")
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
        'results': results
    }

    summary_path = IMAGE_OUTPUT_DIR / 'image_generation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Images saved to: {IMAGE_OUTPUT_DIR}/")

    if len(failed) > 0:
        print(f"\n⚠️  {len(failed)} images failed. You can re-run this script to retry.")


if __name__ == '__main__':
    main()
