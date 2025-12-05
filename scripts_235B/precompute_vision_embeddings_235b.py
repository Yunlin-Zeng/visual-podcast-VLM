#!/usr/bin/env python3
"""
Precompute vision encoder + MLP embeddings for Qwen3-VL-235B training.

This script processes all training images through the frozen vision encoder
and MLP projector, then saves the resulting embeddings to disk. This allows
training the LLM without loading the vision components, saving ~20-35GB per GPU.

Usage:
    python precompute_vision_embeddings_235b.py \
        --model_path /home/ubuntu/LLM/qwen3-vl-235b \
        --data_json /home/ubuntu/image-to-text/Qwen3-VL/data/qwen_training_data_sporc_excerpt.json \
        --output_dir ./precomputed_embeddings_235b_sporc \
        --batch_size 4 \
        --max_samples -1  # -1 for all samples
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import time

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen3VLMoeForConditionalGeneration,
)

def load_model_and_processor(model_path: str, device_map: str = "auto"):
    """Load 235B model with device_map for multi-GPU distribution."""
    print(f"Loading model from {model_path}...")
    print("This will take ~15-20 minutes for 235B model...")

    start_time = time.time()

    # Configure memory limits for 8x A100 GPUs
    num_gpus = torch.cuda.device_count()
    max_memory = {i: "70GB" for i in range(num_gpus)}
    max_memory["cpu"] = "200GB"

    print(f"Using {num_gpus} GPUs with max_memory={max_memory}")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        min_pixels=784,      # 28x28
        max_pixels=50176,    # 224x224
    )

    # Load model with vision encoder + MLP
    # We only need these components, but loading full model is easier
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
    )

    model.eval()  # Set to eval mode (no dropout, etc.)

    elapsed = time.time() - start_time
    print(f"✅ Model loaded in {elapsed/60:.1f} minutes")
    print(f"Model dtype: {model.dtype}")

    return model, processor


def load_dataset(data_json: str) -> List[Dict]:
    """Load training dataset from JSON file."""
    print(f"Loading dataset from {data_json}...")

    with open(data_json, 'r') as f:
        data = json.load(f)

    print(f"✅ Loaded {len(data)} training samples")

    # Count total images
    total_images = sum(len(sample.get('image', [])) for sample in data)
    print(f"✅ Total images to process: {total_images}")

    return data


def resolve_image_paths(image_paths: List[str], data_json_dir: Path) -> List[str]:
    """Resolve relative image paths to absolute paths."""
    resolved = []
    for img_path in image_paths:
        if os.path.isabs(img_path):
            resolved.append(img_path)
        else:
            # Try two strategies:
            # 1. Relative to JSON directory
            abs_path_1 = (data_json_dir / img_path).resolve()
            if abs_path_1.exists():
                resolved.append(str(abs_path_1))
                continue

            # 2. Relative to JSON directory's parent (go up one more level)
            abs_path_2 = (data_json_dir.parent / img_path).resolve()
            if abs_path_2.exists():
                resolved.append(str(abs_path_2))
                continue

            # 3. Relative to current working directory
            abs_path_3 = Path(img_path).resolve()
            if abs_path_3.exists():
                resolved.append(str(abs_path_3))
                continue

            # None worked - use original path and let it fail later
            print(f"⚠️  Could not resolve image path: {img_path}")
            resolved.append(img_path)

    return resolved


def precompute_sample(
    model,
    processor,
    sample: Dict,
    sample_idx: int,
    data_json_dir: Path,
) -> Dict:
    """
    Process one sample through vision encoder + MLP.

    Returns dictionary with:
        - vision_embeddings: [num_images, num_vision_tokens, hidden_dim]
        - vision_grid_thws: [num_images, 3] - spatial grid info for RoPE
        - sample_id: original sample index
        - image_paths: list of image paths processed
    """
    image_paths = sample.get('image', [])
    if not image_paths:
        return None  # Skip text-only samples

    # Resolve image paths
    abs_image_paths = resolve_image_paths(image_paths, data_json_dir)

    # Load images
    images = []
    for img_path in abs_image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"⚠️  Failed to load {img_path}: {e}")
            return None

    if not images:
        return None

    # Process images through vision encoder + MLP
    # We'll use the image_processor directly (not the full processor which expects text)
    with torch.no_grad():
        # Process images using image_processor
        # This returns pixel_values and image_grid_thw
        inputs = processor.image_processor(
            images=images,
            return_tensors="pt",
        )

        # Move to GPU where model's visual encoder is
        pixel_values = inputs['pixel_values']
        image_grid_thw = inputs['image_grid_thw']

        # Determine which device the visual encoder is on
        visual_device = next(model.visual.parameters()).device
        pixel_values = pixel_values.to(visual_device)
        image_grid_thw = image_grid_thw.to(visual_device)

        # Forward through vision encoder
        vision_outputs = model.visual(pixel_values, grid_thw=image_grid_thw)

        # vision_outputs is a tuple - extract the actual embeddings
        # The first element is the vision embeddings tensor
        if isinstance(vision_outputs, tuple):
            vision_embeddings_tensor = vision_outputs[0]
        else:
            vision_embeddings_tensor = vision_outputs

        # vision_embeddings_tensor shape: [total_vision_tokens, hidden_dim]
        # We need to move to CPU and convert to numpy for storage
        # NumPy doesn't support BFloat16, so convert to float32 first, then to float16
        vision_embeddings = vision_embeddings_tensor.cpu().to(torch.float32).numpy().astype(np.float16)
        vision_grid_thws = image_grid_thw.cpu().numpy()

    return {
        'sample_id': sample_idx,
        'vision_embeddings': vision_embeddings,  # [total_tokens, hidden_dim]
        'vision_grid_thws': vision_grid_thws,     # [num_images, 3]
        'image_paths': abs_image_paths,
        'num_images': len(images),
    }


def save_precomputed_sample(
    result: Dict,
    output_dir: Path,
):
    """Save a single precomputed sample."""
    sample_id = result['sample_id']
    sample_file = output_dir / f"sample_{sample_id:04d}.npz"

    # Save all data for this sample
    np.savez_compressed(
        sample_file,
        sample_id=result['sample_id'],
        num_images=result['num_images'],
        image_paths=result['image_paths'],
        vision_embeddings=result['vision_embeddings'],
        vision_grid_thws=result['vision_grid_thws'],
    )


def main():
    parser = argparse.ArgumentParser(description="Precompute vision embeddings for 235B model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to 235B model")
    parser.add_argument("--data_json", type=str, required=True, help="Path to training data JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for embeddings")
    parser.add_argument("--batch_size", type=int, default=4, help="Samples per batch (not GPU batch)")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples to process (-1 for all)")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map for model loading")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Qwen3-VL-235B Vision Embedding Precomputation")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_json}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)

    # Load model and processor
    model, processor = load_model_and_processor(args.model_path, args.device_map)

    # Load dataset
    dataset = load_dataset(args.data_json)
    data_json_dir = Path(args.data_json).parent

    # Limit samples if requested
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]
        print(f"Processing first {args.max_samples} samples only")

    # Process samples
    print("\n" + "=" * 80)
    print("Processing samples...")
    print("=" * 80)

    num_processed = 0
    num_skipped = 0

    for sample_idx, sample in enumerate(tqdm(dataset, desc="Precomputing")):
        result = precompute_sample(
            model=model,
            processor=processor,
            sample=sample,
            sample_idx=sample_idx,
            data_json_dir=data_json_dir,
        )

        if result is None:
            num_skipped += 1
            continue

        # Save each sample immediately
        save_precomputed_sample(result, output_dir)
        num_processed += 1

    # Save metadata
    metadata = {
        'model_path': args.model_path,
        'data_json': args.data_json,
        'num_samples': len(dataset),
        'num_processed': num_processed,
        'num_skipped': num_skipped,
        'processor_config': {
            'min_pixels': processor.image_processor.min_pixels,
            'max_pixels': processor.image_processor.max_pixels,
        },
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ Precomputation complete!")
    print("=" * 80)
    print(f"Total samples: {len(dataset)}")
    print(f"Processed: {num_processed}")
    print(f"Skipped: {num_skipped}")
    print(f"Files saved: {num_processed}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata: {metadata_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
