#!/usr/bin/env python3
"""
Test loading the FP8 quantized model.

FP8 is 8-bit floating point, giving us:
- 222GB total (vs 445GB BF16)
- ~28GB per GPU (vs ~56GB BF16)
- Should fit in 80GB A100s with room for training
"""

import torch
from transformers import Qwen3VLMoeForConditionalGeneration
import os

print("="*80)
print("Testing FP8 Model Loading")
print("="*80)

MODEL_PATH = "/home/ubuntu/LLM/qwen3-vl-235b-fp8"

print(f"\nModel path: {MODEL_PATH}")
print(f"Model size: 222GB (FP8)")
print(f"Expected per-GPU: ~28GB")
print()

try:
    print("Loading model...")

    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float8_e4m3fn,  # FP8 dtype
        trust_remote_code=True,
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
        free = total - reserved
        print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free / {total:.2f}GB total")

    print("\n" + "="*80)
    print("✅ Test PASSED - FP8 Model is ready!")
    print("="*80)

except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*80)
    print("❌ Test FAILED")
    print("="*80)
