"""
Utility functions for Qwen3-VL inference
Contains reusable model loading and inference functions
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from pathlib import Path
import time


def load_model(model_path=None, lora_adapter_path=None):
    """
    Load Qwen3-VL model once and keep in memory

    Args:
        model_path: Path to model weights (default: /home/ubuntu/LLM/qwen3-vl-235b)
        lora_adapter_path: Optional path to LoRA adapter weights to load on top of base model

    Returns:
        tuple: (model, processor)
    """
    # Default to 235B model if not specified
    if model_path is None:
        model_path = "/home/ubuntu/LLM/qwen3-vl-235b"

    print("=" * 80)
    print(f"Loading model from: {model_path}")
    print("=" * 80)

    # Determine GPU memory allocation based on model size
    if "8b" in model_path.lower():
        # 8B model fits on single GPU - force single device to avoid multi-GPU overhead
        max_memory = None
        device_map = {"": "cuda:0"}  # Force single GPU (faster than multi-GPU for 8B)
        print("Loading 8B model (single GPU for optimal speed)...")
    else:
        # 235B model needs multi-GPU - detect available GPUs dynamically
        num_gpus = torch.cuda.device_count()
        max_memory = {i: "70GB" for i in range(num_gpus)}
        device_map = "auto"
        print(f"Loading 235B model with multi-GPU distribution ({num_gpus} GPUs available)...")

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device_map,  # Use conditional device_map
        max_memory=max_memory,
        trust_remote_code=True
    )

    # Load LoRA adapter if provided
    if lora_adapter_path is not None:
        print(f"\nLoading LoRA adapter from: {lora_adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        print("✓ LoRA adapter loaded and attached to base model")

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


def run_inference(model, processor, prompt_text, image_dir=None, image_paths=None, verbose=True, seed=None):
    """
    Run inference with the loaded model

    Args:
        model: Loaded Qwen3-VL model
        processor: Model processor
        prompt_text: Text prompt for generation
        image_dir: Directory containing images (default: story_6228) - deprecated, use image_paths
        image_paths: List of Path objects to 5 images (preferred)
        verbose: Print detailed progress info
        seed: Random seed for reproducibility (default: None = non-deterministic)

    Returns:
        tuple: (output_text, timing_info) or (None, None) on error
    """

    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if verbose:
            print(f"✓ Random seed set to: {seed}")

    start_time = time.time()

    # Handle image paths
    if image_paths is not None:
        # Use provided image paths
        if len(image_paths) == 0:
            if verbose:
                print(f"⚠ Warning: No images provided")
            return None, None
        if verbose:
            print(f"\n✓ Using {len(image_paths)} provided images")
    else:
        # Fall back to image_dir for backward compatibility
        if image_dir is None:
            image_dir = Path("/home/ubuntu/image-to-text/LLaVA-OneVision-1.5/test/story_6228")
        else:
            image_dir = Path(image_dir)

        # Load images from directory (default to 5 for backward compatibility)
        image_paths = sorted(image_dir.glob("image_*.jpg"))[:5]

        if len(image_paths) == 0:
            if verbose:
                print(f"⚠ Warning: No images found in {image_dir}")
            return None, None

        if verbose:
            print(f"\n✓ Using {len(image_paths)} images from: {image_dir.name}")

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
