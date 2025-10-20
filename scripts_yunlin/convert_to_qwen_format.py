"""
Convert our 199 generated samples to Qwen3-VL training format

Input: output_XXX.json files with short_prompt and generated_transcript
Output: qwen_training_data.json in the format:
{
    "image": ["path1.jpg", "path2.jpg", ...],  # 5 images
    "conversations": [
        {"from": "human", "value": "<image><image><image><image><image>\nshort_prompt"},
        {"from": "gpt", "value": "generated_transcript"}
    ]
}
"""

import json
from pathlib import Path
from collections import defaultdict

def load_vist_stories():
    """Load VIST data to get image paths for each story"""

    vist_train_path = Path("/home/ubuntu/image-to-text/data/visual_storytelling/sis/train.story-in-sequence.json")
    vist_images_dir = Path("/home/ubuntu/image-to-text/data/visual_storytelling/images/train")

    print(f"Loading VIST data from {vist_train_path}...")
    with open(vist_train_path, 'r') as f:
        vist_data = json.load(f)

    # Group by story_id
    stories = defaultdict(list)
    for annotation_group in vist_data['annotations']:
        for ann in annotation_group:
            story_id = ann['story_id']
            photo_id = ann['photo_flickr_id']
            order = ann['worker_arranged_photo_order']
            stories[story_id].append({
                'photo_id': photo_id,
                'order': order
            })

    # Get valid stories with exactly 5 images
    valid_stories = []
    for story_id, photos in stories.items():
        if len(photos) == 5:
            photos_sorted = sorted(photos, key=lambda x: x['order'])
            image_ids = [p['photo_id'] for p in photos_sorted]
            image_paths = [str(vist_images_dir / f"{img_id}.jpg") for img_id in image_ids]

            # Verify all images exist
            if all(Path(p).exists() for p in image_paths):
                valid_stories.append({
                    'story_id': story_id,
                    'image_paths': image_paths
                })

        if len(valid_stories) >= 200:
            break

    print(f"Found {len(valid_stories)} valid stories with 5 images each")
    return valid_stories


def convert_samples(outputs_dir, output_file, skip_ids=[25]):
    """Convert all output samples to Qwen training format"""

    outputs_path = Path(outputs_dir)
    valid_stories = load_vist_stories()

    training_data = []
    skipped = []

    print(f"\nConverting samples from {outputs_dir}...")

    for i in range(1, 201):
        if i in skip_ids:
            print(f"  Skipping output_{i:03d}.json (in skip list)")
            skipped.append(i)
            continue

        output_json = outputs_path / f"output_{i:03d}.json"

        if not output_json.exists():
            print(f"  Warning: {output_json.name} not found")
            skipped.append(i)
            continue

        # Load output data
        with open(output_json, 'r') as f:
            data = json.load(f)

        # Get corresponding story images (i-1 because output_001 uses story index 0)
        story_idx = i - 1
        if story_idx >= len(valid_stories):
            print(f"  Warning: No story for output_{i:03d}.json")
            skipped.append(i)
            continue

        story = valid_stories[story_idx]
        image_paths = story['image_paths']

        # Build conversation in Qwen format
        # User message: 5 <image> tags + short prompt text
        user_message = "<image>" * 5 + "\n" + data['short_prompt']

        conversation_entry = {
            "image": image_paths,  # List of 5 image paths
            "conversations": [
                {
                    "from": "human",
                    "value": user_message
                },
                {
                    "from": "gpt",
                    "value": data['generated_transcript']
                }
            ]
        }

        training_data.append(conversation_entry)

    print(f"\nConversion complete:")
    print(f"  Converted: {len(training_data)} samples")
    print(f"  Skipped: {len(skipped)} samples (IDs: {skipped})")

    # Save to JSON file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Print first example for verification
    if training_data:
        print("\n" + "="*80)
        print("First training example (preview):")
        print("="*80)
        first = training_data[0]
        print(f"Images: {len(first['image'])} images")
        print(f"  - {first['image'][0]}")
        print(f"  - {first['image'][1]}")
        print(f"  - ...")
        print(f"\nHuman message (first 200 chars):")
        print(f"  {first['conversations'][0]['value'][:200]}...")
        print(f"\nGPT response (first 200 chars):")
        print(f"  {first['conversations'][1]['value'][:200]}...")
        print("="*80)

    return training_data


if __name__ == "__main__":
    OUTPUTS_DIR = "/home/ubuntu/image-to-text/data/2025-10-16_prompt_templates_tune_qwen3_vl/outputs"
    OUTPUT_FILE = "/home/ubuntu/image-to-text/data/2025-10-16_prompt_templates_tune_qwen3_vl/qwen_training_data.json"

    training_data = convert_samples(OUTPUTS_DIR, OUTPUT_FILE, skip_ids=[25])

    print(f"\n✅ Ready for training with {len(training_data)} samples!")
