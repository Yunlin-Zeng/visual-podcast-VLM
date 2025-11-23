#!/usr/bin/env python3
"""
Convert SPoRC podcast excerpt dataset to Qwen3-VL training format.
Filters to only include excerpts with all images successfully generated.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

# Paths
METADATA_DIR = Path("/home/ubuntu/image-to-text/bedrock_api_pipeline/excerpts_metadata")
IMAGE_DIR = Path("/home/ubuntu/image-to-text/bedrock_api_pipeline/dataset_output")
OUTPUT_FILE = Path("/home/ubuntu/image-to-text/Qwen3-VL/data/qwen_training_data_sporc_excerpt.json")

def load_all_metadata() -> List[Dict[str, Any]]:
    """Load all metadata JSON files."""
    all_excerpts = []

    metadata_files = sorted(METADATA_DIR.glob("ep*_metadata.json"))
    print(f"Found {len(metadata_files)} metadata files")

    for meta_file in metadata_files:
        with open(meta_file, 'r') as f:
            data = json.load(f)
            episode_idx = data['episode_idx']

            for excerpt in data['excerpts']:
                excerpt_data = {
                    'episode_idx': episode_idx,
                    'excerpt_idx': excerpt['excerpt_idx'],
                    'excerpt_text': excerpt['excerpt'],
                    'scenes': excerpt['scenes']
                }
                all_excerpts.append(excerpt_data)

    print(f"Loaded {len(all_excerpts)} total excerpts")
    return all_excerpts

def check_images_exist(excerpt: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Check if all images for an excerpt exist. Returns (all_exist, image_paths)."""
    image_paths = []
    all_exist = True

    for scene in excerpt['scenes']:
        filename = scene['filename']
        image_path = IMAGE_DIR / filename

        if image_path.exists():
            # Use relative path from Qwen3-VL directory
            rel_path = f"../bedrock_api_pipeline/dataset_output/{filename}"
            image_paths.append(rel_path)
        else:
            all_exist = False
            break

    return all_exist, image_paths

def convert_to_qwen_format(excerpts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert excerpts to Qwen3-VL training format."""
    qwen_data = []
    complete_count = 0
    incomplete_count = 0

    for excerpt in excerpts:
        all_exist, image_paths = check_images_exist(excerpt)

        if not all_exist:
            incomplete_count += 1
            continue

        complete_count += 1

        # Build the conversation format
        num_images = len(image_paths)
        image_tags = "<image>" * num_images

        qwen_entry = {
            "image": image_paths,
            "conversations": [
                {
                    "from": "human",
                    "value": f"{image_tags}\nGenerate a natural conversational podcast dialogue. Use the format Speaker 1:, Speaker 2:, Speaker 3:, etc. for multiple speakers. Do not reference the images or use phrases like \"our first image\". Write casual, authentic spoken dialogue without introductions or sign-offs. The word count should be around 800 words."
                },
                {
                    "from": "gpt",
                    "value": excerpt['excerpt_text']
                }
            ]
        }
        qwen_data.append(qwen_entry)

    print(f"\nConversion complete:")
    print(f"  Complete excerpts (all images): {complete_count}")
    print(f"  Incomplete excerpts (missing images): {incomplete_count}")
    print(f"  Success rate: {complete_count / (complete_count + incomplete_count) * 100:.2f}%")

    return qwen_data

def main():
    print("Starting conversion to Qwen3-VL format...")
    print(f"Metadata directory: {METADATA_DIR}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print("-" * 80)

    # Load all metadata
    all_excerpts = load_all_metadata()

    # Convert to Qwen format (filters incomplete excerpts automatically)
    qwen_data = convert_to_qwen_format(all_excerpts)

    # Save to output file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(qwen_data, f, indent=2)

    print(f"\nSaved {len(qwen_data)} training examples to {OUTPUT_FILE}")

    # Show a sample entry
    if qwen_data:
        print("\n" + "=" * 80)
        print("Sample training entry:")
        print("=" * 80)
        sample = qwen_data[0]
        print(f"Number of images: {len(sample['image'])}")
        print(f"Image paths: {sample['image'][:2]}... (showing first 2)")
        print(f"Human prompt: {sample['conversations'][0]['value'][:100]}...")
        print(f"GPT response length: {len(sample['conversations'][1]['value'])} chars")
        print(f"GPT response preview: {sample['conversations'][1]['value'][:200]}...")

if __name__ == "__main__":
    main()
