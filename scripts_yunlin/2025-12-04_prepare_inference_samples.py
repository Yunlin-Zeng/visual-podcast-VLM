"""
Prepare 30 new inference samples from VIST dataset.
Ensures no repeated images across all samples (existing 20 + new 30).
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
import random

# Configuration
VIST_JSON = "/home/ubuntu/image-to-text/data/visual_storytelling/sis/test.story-in-sequence.json"
VIST_IMAGES = "/home/ubuntu/image-to-text/data/visual_storytelling/images/test"
SAMPLES_DIR = "/home/ubuntu/image-to-text/Qwen3-VL/inference_samples"
NUM_NEW_SAMPLES = 30
START_INDEX = 21  # Start from sample_21 (we have 1-20 already)

def get_existing_images():
    """Get all image IDs already used in existing samples."""
    existing_images = set()
    samples_path = Path(SAMPLES_DIR)

    for sample_dir in samples_path.iterdir():
        if sample_dir.is_dir() and sample_dir.name.startswith("sample_"):
            for img_file in sample_dir.glob("image_*.jpg"):
                # Extract image ID from filename like "image_1_4379270527.jpg"
                parts = img_file.stem.split("_")
                if len(parts) >= 3:
                    img_id = parts[2]
                    existing_images.add(img_id)

    return existing_images

def get_existing_story_ids():
    """Get story IDs already used in existing samples."""
    existing_stories = set()
    samples_path = Path(SAMPLES_DIR)

    for sample_dir in samples_path.iterdir():
        if sample_dir.is_dir() and sample_dir.name.startswith("sample_"):
            # Extract story_id from directory name like "sample_01_story_48269"
            parts = sample_dir.name.split("_")
            if len(parts) >= 4 and parts[2] == "story":
                existing_stories.add(parts[3])

    return existing_stories

def load_vist_data():
    """Load and process VIST annotations."""
    with open(VIST_JSON) as f:
        data = json.load(f)

    # Group annotations by story_id
    stories = defaultdict(list)
    for ann_group in data["annotations"]:
        for ann in ann_group:
            story_id = ann["story_id"]
            stories[story_id].append(ann)

    # Sort each story's annotations by photo order
    for story_id in stories:
        stories[story_id].sort(key=lambda x: x["worker_arranged_photo_order"])

    return stories

def select_new_samples(stories, existing_images, existing_story_ids, num_samples):
    """Select new samples with no repeated images."""
    selected = []
    used_images = existing_images.copy()

    # Get candidate stories (not already used)
    candidates = [
        (story_id, anns) for story_id, anns in stories.items()
        if story_id not in existing_story_ids and len(anns) == 5
    ]

    # Shuffle for randomness
    random.seed(42)  # For reproducibility
    random.shuffle(candidates)

    for story_id, anns in candidates:
        if len(selected) >= num_samples:
            break

        # Get image IDs for this story
        story_images = [ann["photo_flickr_id"] for ann in anns]

        # Check if any image is already used
        if any(img_id in used_images for img_id in story_images):
            continue

        # Check if images exist on disk
        images_exist = all(
            Path(VIST_IMAGES) / f"{img_id}.jpg" in list(Path(VIST_IMAGES).glob("*.jpg")) or
            (Path(VIST_IMAGES) / f"{img_id}.jpg").exists()
            for img_id in story_images
        )

        if not images_exist:
            # Double check by actually looking for files
            missing = []
            for img_id in story_images:
                img_path = Path(VIST_IMAGES) / f"{img_id}.jpg"
                if not img_path.exists():
                    missing.append(img_id)
            if missing:
                print(f"  Skipping story {story_id}: missing images {missing}")
                continue

        # This story is valid - add it
        selected.append((story_id, anns))
        used_images.update(story_images)
        print(f"  Selected story {story_id} with images: {story_images}")

    return selected

def create_sample_directories(selected_samples, start_idx):
    """Create sample directories with images and ground truth."""
    for i, (story_id, anns) in enumerate(selected_samples):
        sample_num = start_idx + i
        sample_name = f"sample_{sample_num:02d}_story_{story_id}"
        sample_dir = Path(SAMPLES_DIR) / sample_name

        # Create directory
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        for j, ann in enumerate(anns, 1):
            img_id = ann["photo_flickr_id"]
            src = Path(VIST_IMAGES) / f"{img_id}.jpg"
            dst = sample_dir / f"image_{j}_{img_id}.jpg"
            shutil.copy2(src, dst)

        # Create ground truth story file
        story_text = " ".join(ann["text"] for ann in anns)
        with open(sample_dir / "ground_truth_story.txt", "w") as f:
            f.write(story_text)

        print(f"Created {sample_name}")

def main():
    print("=" * 60)
    print("Preparing 30 new inference samples from VIST dataset")
    print("=" * 60)

    # Get existing data
    print("\n1. Checking existing samples...")
    existing_images = get_existing_images()
    existing_story_ids = get_existing_story_ids()
    print(f"   Found {len(existing_story_ids)} existing stories")
    print(f"   Found {len(existing_images)} existing images")
    print(f"   Existing story IDs: {sorted(existing_story_ids)}")

    # Load VIST data
    print("\n2. Loading VIST data...")
    stories = load_vist_data()
    print(f"   Found {len(stories)} unique stories in VIST test set")

    # Select new samples
    print(f"\n3. Selecting {NUM_NEW_SAMPLES} new samples with unique images...")
    selected = select_new_samples(stories, existing_images, existing_story_ids, NUM_NEW_SAMPLES)
    print(f"   Selected {len(selected)} new samples")

    if len(selected) < NUM_NEW_SAMPLES:
        print(f"\n   WARNING: Could only find {len(selected)} valid samples!")

    # Create directories
    print(f"\n4. Creating sample directories (starting from sample_{START_INDEX:02d})...")
    create_sample_directories(selected, START_INDEX)

    print("\n" + "=" * 60)
    print(f"Done! Created {len(selected)} new sample directories")
    print(f"Total samples now: {len(existing_story_ids) + len(selected)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
