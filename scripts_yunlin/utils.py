"""
Utility functions for Qwen3-VL inference
Contains reusable model loading and inference functions
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from pathlib import Path
import time


def load_model():
    """Load Qwen3-VL model once and keep in memory"""
    print("=" * 80)
    print("Loading Qwen3-VL-235B-A22B-Instruct model...")
    print("=" * 80)

    model_path = "/home/ubuntu/LLM/qwen3-vl-235b"

    # Force distribution across all 8 GPUs using max_memory
    max_memory = {i: "70GB" for i in range(8)}

    print("Loading model with multi-GPU distribution...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,  # Force model splitting across 8 GPUs
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Set image resolution - balance quality vs memory
    # 512×32×32 = ~524k pixels per image, ~2k tokens per image, ~10k total for 5 images
    # Using lower resolution to accommodate longer system prompts
    processor.image_processor.size = {
        "longest_edge": 512 * 32 * 32,  # 524,288 pixels max per image
        "shortest_edge": 256 * 32 * 32   # 262,144 pixels min per image
    }

    print(f"\n✓ Model loaded successfully!")
    print(f"✓ Model distributed across GPUs")
    print(f"✓ dtype: {model.dtype}")
    print(f"✓ Image resolution: 512×32×32 (~524k pixels max per image)")

    return model, processor


def run_inference(model, processor, prompt_text, image_dir=None, image_paths=None, verbose=True):
    """
    Run inference with the loaded model

    Args:
        model: Loaded Qwen3-VL model
        processor: Model processor
        prompt_text: Text prompt for generation
        image_dir: Directory containing images (default: story_6228) - deprecated, use image_paths
        image_paths: List of Path objects to 5 images (preferred)
        verbose: Print detailed progress info

    Returns:
        tuple: (output_text, timing_info) or (None, None) on error
    """

    start_time = time.time()

    # Handle image paths
    if image_paths is not None:
        # Use provided image paths
        if len(image_paths) != 5:
            if verbose:
                print(f"⚠ Warning: Provided {len(image_paths)} images, expected 5")
            return None, None
        if verbose:
            print(f"\n✓ Using {len(image_paths)} provided images")
    else:
        # Fall back to image_dir for backward compatibility
        if image_dir is None:
            image_dir = Path("/home/ubuntu/image-to-text/LLaVA-OneVision-1.5/test/story_6228")
        else:
            image_dir = Path(image_dir)

        # Load images from directory
        image_paths = sorted(image_dir.glob("image_*.jpg"))[:5]

        if len(image_paths) != 5:
            if verbose:
                print(f"⚠ Warning: Found {len(image_paths)} images, expected 5")
            return None, None

        if verbose:
            print(f"\n✓ Using images from: {image_dir.name}")

    # Create messages - use plain string paths for local files
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_paths[0])},
                {"type": "image", "image": str(image_paths[1])},
                {"type": "image", "image": str(image_paths[2])},
                {"type": "image", "image": str(image_paths[3])},
                {"type": "image", "image": str(image_paths[4])},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    # Prepare inputs using official API from README
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    prep_time = time.time()
    if verbose:
        print(f"✓ Input tokens: {inputs['input_ids'].shape[1]}")
        print(f"✓ Input preparation time: {prep_time - start_time:.2f}s")
        print("\n🚀 Generating...")

    # Generate - with safety limit to prevent OOM
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,  # Safety limit while allowing long transcripts
            do_sample=True,
            temperature=0.7,
            top_p=0.8
        )

    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    gen_time = time.time()
    total_time = gen_time - start_time
    generation_time = gen_time - prep_time
    num_tokens = len(generated_ids_trimmed[0])
    tokens_per_sec = num_tokens / generation_time if generation_time > 0 else 0

    timing_info = {
        "input_prep_time": prep_time - start_time,
        "generation_time": generation_time,
        "total_time": total_time,
        "num_tokens": num_tokens,
        "tokens_per_sec": tokens_per_sec
    }

    if verbose:
        print(f"\n⏱️  Timing:")
        print(f"   - Input preparation: {timing_info['input_prep_time']:.2f}s")
        print(f"   - Generation: {timing_info['generation_time']:.2f}s")
        print(f"   - Total: {timing_info['total_time']:.2f}s")
        print(f"   - Generated {num_tokens} tokens at {tokens_per_sec:.2f} tokens/s")

    # Clear GPU cache to prevent memory accumulation across multiple inferences
    torch.cuda.empty_cache()

    return output_text[0], timing_info
