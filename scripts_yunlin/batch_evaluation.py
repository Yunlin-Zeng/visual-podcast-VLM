#!/usr/bin/env python3
"""
Batch evaluation script to compare base model vs fine-tuned checkpoint-56
on 20 unique VIST samples.
"""

import os
import sys
import json
import torch
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
import random

# Add parent directory to path for utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_model

# Detailed Q&A prompt (Test 1 format - what was used for training)
PROMPT_TEMPLATE = """<image><image><image><image><image>
You are creating a podcast transcript in a Q&A format between two hosts based on the image I uploaded. They narrate a descriptive and detailed story based on the images. They can ask questions. Based on the visual information from these 5 images, create an engaging conversational podcast transcript.

CRITICAL REQUIREMENTS:

1. STRUCTURE (REQUIRED):

* One host guides the talk. Sometimes either host can ask questions.
* Both hosts can provide rich visual descriptions
* Questions should follow the story arc: setting → characters (if any) → action → details → conclusion

   Question pattern example:

* Q1: "What's happening here? Where does this take place?"
* Q2: "Tell me about the people involved. What are they wearing?"
* Q3: "What happens next in the story?"
* Q4: "Are there any other interesting details?"
* Q5: "How does this story end?"

2. FORMAT (NON-NEGOTIABLE):

* Do NOT cut the conversation short
* Each answer should be substantial

3. VISUAL DETAILS IN ANSWERS (REQUIRED):

   Must include SPECIFIC, RICH visual descriptions:

   ✓ GOOD examples (be this specific):

* "black jacket with bright yellow racing stripes down the sleeves"
* "light blue plaid boxer shorts with white cross-hatching pattern"
* "lace curtains filtering soft morning light"
* "birthday card with 'HAPPY BIRTHDAY' written in bold red marker"
* "wooden table with warm honey-colored finish"

   ✗ BAD examples (avoid generic descriptions):

* "nice jacket" (too vague)
* "blue boxers" (missing pattern detail)
* "pretty kitchen" (no specific details)

   Details to include in answers:

* Exact colors and patterns
* Specific clothing details and materials
* Textures and finishes
* Environmental elements (decorations, furniture, wall items)
* Spatial relationships
* Lighting and atmosphere

4. NATURAL DIALOGUE (REQUIRED):

* DO NOT mention "images", "photos", "pictures"
* Tell the story naturally
* Questions should feel curious and engaged
* Answers should be descriptive and informative

   ✓ GOOD:
   What's happening in this birthday scene? Where does this take place?
   This unfolds in the coziest home kitchen. We've got delicate white lace curtains filtering natural light, walls covered with family photos and postcards, even a heart decoration made from blue handprints.

   ✗ BAD:
   What do you see in the first image?
   In the image, there is a kitchen with curtains.


5. (optional) STORY PROGRESSION THROUGH QUESTIONS:

* Each answer should build on previous information

Please generate a transcript in a conversational style, with a dramatic tone, in podcast format, with length between 600-700 words (do NOT exceed 700 words), and elaborate detail level.

Generate a two-host conversation about this visual sequence."""


def load_vist_test_stories(annotation_file, image_dir, num_samples=20):
    """Load VIST test stories with unique images."""
    print(f"  Loading VIST annotations from {annotation_file}...")
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Parse stories
    annotations = data['annotations']

    # Group by story_id
    stories = {}
    for ann in annotations:
        story_id = ann[0]['story_id']
        if story_id not in stories:
            stories[story_id] = []
        stories[story_id].append(ann[0])

    # Filter stories with exactly 5 images and sort by order
    complete_stories = []
    for story_id, images in stories.items():
        if len(images) == 5:
            # Sort by worker_arranged_photo_order
            images_sorted = sorted(images, key=lambda x: x['worker_arranged_photo_order'])
            complete_stories.append({
                'story_id': story_id,
                'images': images_sorted
            })

    print(f"  Found {len(complete_stories)} complete stories with 5 images")

    # Select stories with unique images (greedy approach)
    selected_stories = []
    used_image_ids = set()

    # Random shuffle for variety
    random.shuffle(complete_stories)

    for story in complete_stories:
        if len(selected_stories) >= num_samples:
            break

        # Get image IDs for this story
        image_ids = [img['photo_flickr_id'] for img in story['images']]

        # Check if any images already used
        overlap = set(image_ids).intersection(used_image_ids)

        if not overlap:  # No overlap - prefer these
            selected_stories.append(story)
            used_image_ids.update(image_ids)

    # If we still need more, allow some overlap
    if len(selected_stories) < num_samples:
        for story in complete_stories:
            if len(selected_stories) >= num_samples:
                break
            if story not in selected_stories:
                selected_stories.append(story)

    # Convert to image paths
    result = []
    for i, story in enumerate(selected_stories[:num_samples], 1):
        image_paths = []
        for img in story['images']:
            img_id = img['photo_flickr_id']
            img_path = Path(image_dir) / f"{img_id}.jpg"
            if img_path.exists():
                image_paths.append(img_path)
            else:
                print(f"    Warning: Image {img_id}.jpg not found")

        if len(image_paths) == 5:
            result.append({
                'id': i,
                'story_id': story['story_id'],
                'images': image_paths
            })

    print(f"  ✓ Selected {len(result)} stories with all images available")
    return result


def run_inference(model, processor, images, prompt, seed=42):
    """Run inference on a set of images."""
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(img)} for img in images
            ] + [{"type": "text", "text": prompt}]
        }
    ]

    # Process
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False  # Greedy decoding for consistency
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


def main():
    print("="*80)
    print("Batch Evaluation: Base Model vs Checkpoint-56")
    print("="*80)

    # Set random seed for reproducibility
    random.seed(42)

    # Paths
    vist_annotation = "/home/ubuntu/image-to-text/data/visual_storytelling/sis/test.story-in-sequence.json"
    vist_images = "/home/ubuntu/image-to-text/data/visual_storytelling/images/test"
    base_model_path = "/home/ubuntu/LLM/qwen3-vl-8b"
    checkpoint_path = "/home/ubuntu/image-to-text/Qwen3-VL/finetuned_models/2025-11-10_qwen3vl-8b-sporc-v2-conservative/checkpoint-56"

    # Output directories
    base_output_dir = Path("/home/ubuntu/LLM/qwen3-vl-8b/evaluation_outputs")
    checkpoint_output_dir = Path(checkpoint_path) / "evaluation_outputs"
    base_output_dir.mkdir(exist_ok=True)
    checkpoint_output_dir.mkdir(exist_ok=True)

    # Find 20 unique samples
    print("\n[1/4] Loading 20 unique VIST test stories...")
    stories = load_vist_test_stories(vist_annotation, vist_images, num_samples=20)

    if len(stories) < 20:
        print(f"Warning: Only found {len(stories)} stories, proceeding with available samples")

    # Save sample list
    sample_list = []
    for story in stories:
        sample_list.append({
            'id': story['id'],
            'story_id': story['story_id'],
            'images': [str(img) for img in story['images']]
        })

    with open(base_output_dir / "sample_list.json", 'w') as f:
        json.dump(sample_list, f, indent=2)
    with open(checkpoint_output_dir / "sample_list.json", 'w') as f:
        json.dump(sample_list, f, indent=2)

    print(f"✓ Sample list saved")

    # Load base model
    print("\n[2/4] Loading base model...")
    base_model, processor = load_model(base_model_path, lora_adapter_path=None)
    print("✓ Base model loaded")

    # Run base model inference
    print(f"\n[3/4] Running base model inference on {len(stories)} samples...")
    for story in stories:
        print(f"  Sample {story['id']}/{len(stories)}: story_{story['story_id']}...", end=" ", flush=True)

        output = run_inference(base_model, processor, story['images'], PROMPT_TEMPLATE, seed=42)

        # Save output
        output_file = base_output_dir / f"sample_{story['id']:02d}_story_{story['story_id']}.txt"
        with open(output_file, 'w') as f:
            f.write(f"Story ID: {story['story_id']}\n")
            f.write(f"Images:\n")
            for img in story['images']:
                f.write(f"  - {img.name}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"PROMPT:\n")
            f.write(f"{'='*80}\n")
            f.write(PROMPT_TEMPLATE)
            f.write(f"\n\n{'='*80}\n")
            f.write(f"OUTPUT:\n")
            f.write(f"{'='*80}\n")
            f.write(output)
            f.write(f"\n\n{'='*80}\n")
            f.write(f"Word count: {len(output.split())} words\n")

        print(f"✓ ({len(output.split())} words)")

    # Clean up base model
    del base_model
    torch.cuda.empty_cache()

    # Load fine-tuned model
    print(f"\n[4/4] Loading checkpoint-56 and running inference on {len(stories)} samples...")
    finetuned_model, processor = load_model(base_model_path, lora_adapter_path=checkpoint_path)
    print("✓ Fine-tuned model loaded")

    for story in stories:
        print(f"  Sample {story['id']}/{len(stories)}: story_{story['story_id']}...", end=" ", flush=True)

        output = run_inference(finetuned_model, processor, story['images'], PROMPT_TEMPLATE, seed=42)

        # Save output
        output_file = checkpoint_output_dir / f"sample_{story['id']:02d}_story_{story['story_id']}.txt"
        with open(output_file, 'w') as f:
            f.write(f"Story ID: {story['story_id']}\n")
            f.write(f"Images:\n")
            for img in story['images']:
                f.write(f"  - {img.name}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"PROMPT:\n")
            f.write(f"{'='*80}\n")
            f.write(PROMPT_TEMPLATE)
            f.write(f"\n\n{'='*80}\n")
            f.write(f"OUTPUT:\n")
            f.write(f"{'='*80}\n")
            f.write(output)
            f.write(f"\n\n{'='*80}\n")
            f.write(f"Word count: {len(output.split())} words\n")

        print(f"✓ ({len(output.split())} words)")

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(f"Base model outputs: {base_output_dir}")
    print(f"Checkpoint-56 outputs: {checkpoint_output_dir}")
    print("\nNext steps:")
    print("  1. Review outputs for quality comparison")
    print("  2. Check for repetition loops")
    print("  3. Evaluate emotional engagement and thematic coherence")


if __name__ == "__main__":
    main()
