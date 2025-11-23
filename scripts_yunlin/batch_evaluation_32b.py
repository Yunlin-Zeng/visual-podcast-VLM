#!/usr/bin/env python3
"""
Batch evaluation script for Qwen3-VL-32B model variants.

Evaluates 4 model variants on the same 20 VIST test samples:
1. Base model (no fine-tuning)
2. Checkpoint-56 (epoch 1)
3. Checkpoint-112 (epoch 2)
4. Checkpoint-168 (epoch 3, final)

Uses the same evaluation setup as the 8B model comparison.
"""

import json
import random
import torch
from pathlib import Path
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# Training prompt template - MUST match training data exactly
PROMPT_TEMPLATE = """<image><image><image><image><image>
Generate a natural conversational podcast dialogue. Use the format Speaker 1:, Speaker 2:, Speaker 3:, etc. for multiple speakers. Do not reference the images or use phrases like "our first image". Write casual, authentic spoken dialogue without introductions or sign-offs. The word count should be around 800 words.
"""


def load_vist_test_stories(annotation_file, image_dir, num_samples=20):
    """Load VIST test stories with unique images (same as 8B evaluation)."""
    print(f"Loading VIST test stories from {annotation_file}")

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

    # Random shuffle for variety (use same seed as 8B evaluation)
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


def load_model_and_processor(model_path, lora_adapter_path=None):
    """Load model with optional LoRA adapter."""
    print(f"\nLoading model from: {model_path}")
    if lora_adapter_path:
        print(f"  + LoRA adapter: {lora_adapter_path}")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_path)

    # Load base model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    # Load LoRA adapter if specified
    if lora_adapter_path:
        model = PeftModel.from_pretrained(model, lora_adapter_path)

    model.eval()
    print(f"  ✓ Model loaded successfully")

    return model, processor


def main():
    import argparse
    import re

    parser = argparse.ArgumentParser(description="Batch evaluation for Qwen3-VL-32B checkpoints")
    parser.add_argument("--lora-dir", type=str, required=True,
                        help="Directory containing LoRA checkpoints (e.g., finetuned_models/2025-11-22_qwen3vl-32b-sporc-lora)")
    parser.add_argument("--include-base", action="store_true", default=True,
                        help="Also evaluate base model without LoRA")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of test samples to evaluate")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)

    # Paths
    vist_annotation = "/home/ubuntu/image-to-text/data/visual_storytelling/sis/test.story-in-sequence.json"
    vist_images = "/home/ubuntu/image-to-text/data/visual_storytelling/images/test"
    base_model_path = "/home/ubuntu/LLM/qwen3-vl-32b"
    lora_dir = Path(args.lora_dir)

    # Dynamically discover checkpoints
    checkpoint_dirs = sorted(
        [d for d in lora_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(re.search(r"checkpoint-(\d+)", x.name).group(1))
    )

    # Build model variants list
    model_variants = []

    # Optionally add base model
    if args.include_base:
        model_variants.append({
            "name": "base",
            "description": "Base model (no fine-tuning)",
            "model_path": base_model_path,
            "lora_adapter": None
        })

    # Add all discovered checkpoints
    for ckpt_dir in checkpoint_dirs:
        ckpt_num = re.search(r"checkpoint-(\d+)", ckpt_dir.name).group(1)
        model_variants.append({
            "name": ckpt_dir.name,
            "description": f"Checkpoint-{ckpt_num}",
            "model_path": base_model_path,
            "lora_adapter": str(ckpt_dir)
        })

    print("="*80)
    print("Batch Evaluation: Qwen3-VL-32B Model Variants Comparison")
    print("="*80)
    print(f"\nLoRA directory: {lora_dir}")
    print(f"Found {len(checkpoint_dirs)} checkpoints:")
    for i, variant in enumerate(model_variants, 1):
        print(f"  {i}. {variant['description']}")
    print("="*80)

    # Load test samples
    print("\n" + "="*80)
    test_stories = load_vist_test_stories(vist_annotation, vist_images, num_samples=args.num_samples)
    print("="*80)

    # Evaluate each model variant
    for variant in model_variants:
        print(f"\n{'='*80}")
        print(f"Evaluating: {variant['description']}")
        print(f"{'='*80}")

        # Load model
        model, processor = load_model_and_processor(
            variant['model_path'],
            variant['lora_adapter']
        )

        # Create output directory
        if variant['lora_adapter']:
            output_dir = Path(variant['lora_adapter']) / "evaluation_outputs_corrected_prompt"
        else:
            output_dir = Path(variant['model_path']) / "evaluation_outputs_32b_base_corrected_prompt"
        output_dir.mkdir(exist_ok=True, parents=True)

        # Run inference on all test samples
        print(f"\nRunning inference on {len(test_stories)} samples...")
        for i, story in enumerate(test_stories, 1):
            print(f"  [{i:2d}/{len(test_stories)}] Story {story['story_id']}...", end=" ", flush=True)

            try:
                # Run inference
                output = run_inference(model, processor, story['images'], PROMPT_TEMPLATE, seed=42)

                # Save output
                output_file = output_dir / f"sample_{i:02d}_story_{story['story_id']}.txt"
                with open(output_file, 'w') as f:
                    f.write(f"Story ID: {story['story_id']}\n")
                    f.write(f"Sample: {i}/20\n")
                    f.write(f"\nModel: {variant['description']}\n")
                    f.write("="*80 + "\n\n")
                    f.write(output)

                # Count words
                word_count = len(output.split())
                print(f"✓ ({word_count} words)")

            except Exception as e:
                print(f"✗ Error: {e}")
                continue

        print(f"\n✓ Outputs saved to: {output_dir}")

        # Clean up model to free memory
        del model
        torch.cuda.empty_cache()
        print("  ✓ Model unloaded, memory freed")

    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)
    print("\nOutput locations:")
    for variant in model_variants:
        if variant['lora_adapter']:
            output_dir = Path(variant['lora_adapter']) / "evaluation_outputs_corrected_prompt"
        else:
            output_dir = Path(variant['model_path']) / "evaluation_outputs_32b_base_corrected_prompt"
        print(f"  {variant['name']:20s}: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
