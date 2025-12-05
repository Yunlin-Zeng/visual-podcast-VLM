#!/usr/bin/env python3
"""
Test loading model with CPU offloading strategy.

Strategy:
1. Load BF16 model to CPU (fits in 1.1TB RAM)
2. Apply 4-bit quantization on CPU
3. Move quantized model to GPU (should be ~15GB/GPU)
"""

import torch
from transformers import Qwen3VLMoeForConditionalGeneration, BitsAndBytesConfig
import os

print("="*80)
print("Testing CPU-First Loading with Quantization")
print("="*80)

MODEL_PATH = "/home/ubuntu/LLM/qwen3-vl-235b-a22b"

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print(f"\nModel path: {MODEL_PATH}")
print(f"Quantization config: NF4, compute_dtype=bfloat16, double_quant=True")
print(f"\nStrategy: Load to CPU first, quantize, then dispatch to GPU")
print(f"Available RAM: 1.1TB (enough for 445GB BF16 model)")
print()

try:
    print("Loading model...")
    print("  This will load the BF16 model to CPU, quantize it, then move to GPU")
    print("  Expected behavior: High CPU RAM usage initially, then quantized model on GPU")
    print()

    # Use max_memory to control device placement
    # Start with CPU only, let accelerate handle the rest
    max_memory = {i: "79GiB" for i in range(8)}
    max_memory["cpu"] = "800GiB"  # Use CPU RAM for overflow

    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        low_cpu_mem_usage=False,  # Allow using CPU RAM freely
    )

    print("\n✅ Model loaded successfully!")
    print(f"Model class: {model.__class__.__name__}")

    print("\nDevice map (first 30 layers):")
    if hasattr(model, 'hf_device_map'):
        for key, device in sorted(model.hf_device_map.items())[:30]:
            print(f"  {key}: {device}")
        if len(model.hf_device_map) > 30:
            print(f"  ... and {len(model.hf_device_map) - 30} more layers")

    # Check GPU memory usage
    print("\nGPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved / {total:.2f}GB total")

    print("\n" + "="*80)
    print("✅ Test PASSED - Model is ready for training!")
    print("="*80)

except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*80)
    print("❌ Test FAILED")
    print("="*80)
