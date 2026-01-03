#!/usr/bin/env python3
"""
Test loading the fixed 4-bit quantized model.
"""

import torch
from transformers import Qwen3VLMoeForConditionalGeneration
import os

print("="*80)
print("Testing Fixed 4-bit Quantized Model Load")
print("="*80)

MODEL_PATH = "/home/ubuntu/LLM/qwen3-vl-235b-4bit-fixed"

print(f"\nModel path: {MODEL_PATH}")
print(f"Loading model...")

try:
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    print("\n✅ Model loaded successfully!")
    print(f"Model class: {model.__class__.__name__}")
    print(f"\nDevice map:")
    if hasattr(model, 'hf_device_map'):
        for key, device in sorted(model.hf_device_map.items())[:20]:
            print(f"  {key}: {device}")
        if len(model.hf_device_map) > 20:
            print(f"  ... and {len(model.hf_device_map) - 20} more layers")

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
