#!/usr/bin/env python3
"""
Focused evaluation for checkpoint-56 only (32B epoch 1).
Runs all 20 samples with timeout protection.
"""

import json
import random
import torch
import sys
from pathlib import Path
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import signal
from contextlib import contextmanager

# Timeout handler
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Detailed Q&A prompt template
PROMPT_TEMPLATE = """<image><image><image><image><image>

You are creating a podcast transcript in a Q&A format between two hosts discussing a photo series. Follow these requirements:

CRITICAL REQUIREMENTS:

1. STRUCTURE (REQUIRED):
   - Question and Answer (Q&A) format between two hosts
   - Questions must be SHORT (1-2 sentences max)
   - Answers must be SUBSTANTIAL (4-8 sentences)
   - Must include rich visual descriptions in the ANSWERS

2. FORMAT (NON-NEGOTIABLE):
   - Start: "**Question:** [short question]"
   - Follow: "**Answer:** [substantial answer with visual details]"
   - Each Q&A pair separated by blank line
   - NO colons after "Question" or "Answer"

3. VISUAL DETAILS IN ANSWERS (REQUIRED):
   - Describe specific colors, patterns, textures
   - Mention backgrounds and settings
   - Note expressions, gestures, compositions
   - These details must appear IN THE ANSWERS, not questions

4. NATURAL DIALOGUE (REQUIRED):
   - Never say "in the first image", "in the second photo"
   - Instead: "at the beginning", "later", "as the story unfolds"
   - NO MENTION of "images", "photos", or "pictures"
   - Talk about the story/scene/moment naturally

5. STORY PROGRESSION THROUGH QUESTIONS:
   - Question 1: Opening scene/setting
   - Question 2-3: Development/middle parts
   - Question 4: Resolution/conclusion
   - Questions guide the narrative flow

Generate 600-700 words total. Do NOT exceed 700 words.
"""


def load_vist_test_stories(annotation_file, image_dir, num_samples=20):
    """Load VIST test stories with unique images."""
    print(f"Loading VIST test stories from {annotation_file}", flush=True)

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
            images_sorted = sorted(images, key=lambda x: x['worker_arranged_photo_order'])
            complete_stories.append({
                'story_id': story_id,
                'images': images_sorted
            })

    print(f"  Found {len(complete_stories)} complete stories with 5 images", flush=True)

    # Select stories with unique images (same seed as original evaluation)
    selected_stories = []
    used_image_ids = set()
    random.shuffle(complete_stories)

    for story in complete_stories:
        if len(selected_stories) >= num_samples:
            break

        image_ids = [img['photo_flickr_id'] for img in story['images']]
        overlap = set(image_ids).intersection(used_image_ids)

        if not overlap:
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
                print(f"    Warning: Image {img_id}.jpg not found", flush=True)

        if len(image_paths) == 5:
            result.append({
                'id': i,
                'story_id': story['story_id'],
                'images': image_paths
            })

    print(f"  ✓ Selected {len(result)} stories with all images available", flush=True)
    return result


def run_inference(model, processor, images, prompt, seed=42, max_time=300):
    """Run inference with timeout protection (5 minutes per sample)."""
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

    # Generate with timeout
    try:
        with time_limit(max_time):
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False  # Greedy decoding for consistency
                )
    except TimeoutException:
        return f"[TIMEOUT after {max_time} seconds]"

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
    print("="*80, flush=True)
    print("Checkpoint-56 Evaluation: 32B Epoch 1 Fine-tuned Model", flush=True)
    print("="*80, flush=True)

    # Set random seed (same as original evaluation)
    random.seed(42)

    # Paths
    vist_annotation = "/home/ubuntu/image-to-text/data/visual_storytelling/sis/test.story-in-sequence.json"
    vist_images = "/home/ubuntu/image-to-text/data/visual_storytelling/images/test"
    base_model_path = "/home/ubuntu/LLM/qwen3-vl-32b"
    lora_adapter = "./finetuned_models/2025-11-12_qwen3vl-32b-sporc-lora/checkpoint-56"

    # Output directory
    output_dir = Path(lora_adapter) / "evaluation_outputs"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load 20 test samples
    print("\n" + "="*80, flush=True)
    test_stories = load_vist_test_stories(vist_annotation, vist_images, num_samples=20)
    print("="*80, flush=True)

    # Check which samples are already completed
    existing_outputs = set()
    for f in output_dir.glob("sample_*_story_*.txt"):
        existing_outputs.add(f.stem)

    print(f"\nFound {len(existing_outputs)} existing outputs, will skip those", flush=True)

    # Load model
    print(f"\nLoading model from: {base_model_path}", flush=True)
    print(f"  + LoRA adapter: {lora_adapter}", flush=True)

    processor = AutoProcessor.from_pretrained(base_model_path)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    model = PeftModel.from_pretrained(model, lora_adapter)
    model.eval()
    print(f"  ✓ Model loaded successfully", flush=True)

    # Run inference on all test samples
    print(f"\nRunning inference on {len(test_stories)} samples...", flush=True)
    print(f"(Skipping {len(existing_outputs)} already completed)", flush=True)

    for i, story in enumerate(test_stories, 1):
        output_file = output_dir / f"sample_{i:02d}_story_{story['story_id']}.txt"

        # Skip if already exists
        if output_file.stem in existing_outputs:
            print(f"  [{i:2d}/{len(test_stories)}] Story {story['story_id']}... ⏭ (already exists)", flush=True)
            continue

        print(f"  [{i:2d}/{len(test_stories)}] Story {story['story_id']}...", end=" ", flush=True)

        try:
            # Run inference with 5-minute timeout
            output = run_inference(model, processor, story['images'], PROMPT_TEMPLATE, seed=42, max_time=300)

            # Check if timed out
            if "[TIMEOUT" in output:
                print(f"⏱ TIMEOUT (skipped)", flush=True)
                # Save timeout marker
                with open(output_file, 'w') as f:
                    f.write(f"Story ID: {story['story_id']}\n")
                    f.write(f"Sample: {i}/20\n")
                    f.write(f"\nModel: Checkpoint-56 (epoch 1)\n")
                    f.write("="*80 + "\n\n")
                    f.write(output)
                continue

            # Save output
            with open(output_file, 'w') as f:
                f.write(f"Story ID: {story['story_id']}\n")
                f.write(f"Sample: {i}/20\n")
                f.write(f"\nModel: Checkpoint-56 (epoch 1)\n")
                f.write("="*80 + "\n\n")
                f.write(output)

            # Count words
            word_count = len(output.split())
            print(f"✓ ({word_count} words)", flush=True)

        except Exception as e:
            print(f"✗ Error: {e}", flush=True)
            # Save error
            with open(output_file, 'w') as f:
                f.write(f"Story ID: {story['story_id']}\n")
                f.write(f"Sample: {i}/20\n")
                f.write(f"\nModel: Checkpoint-56 (epoch 1)\n")
                f.write("="*80 + "\n\n")
                f.write(f"[ERROR: {str(e)}]")
            continue

    print(f"\n✓ Outputs saved to: {output_dir}", flush=True)
    print("="*80, flush=True)
    print("Evaluation complete!", flush=True)
    print("="*80, flush=True)


if __name__ == "__main__":
    main()
