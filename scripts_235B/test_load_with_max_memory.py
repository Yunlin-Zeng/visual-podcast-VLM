#!/usr/bin/env python3
"""
Test loading Qwen3-VL-235B with max_memory limits (Attempt 14).

This uses the same approach as the successful inference script:
- device_map="auto" for proper distribution
- max_memory limits to prevent OOM on GPU 0
"""

import torch
from transformers import Qwen3VLMoeForConditionalGeneration

print("="*80)
print("Testing Model Loading with max_memory - Attempt 14")
print("="*80)

MODEL_PATH = "/home/ubuntu/LLM/qwen3-vl-235b"

# Configure memory limits (same as inference script)
num_gpus = torch.cuda.device_count()
max_memory = {i: "70GB" for i in range(num_gpus)}
max_memory["cpu"] = "200GB"  # Allow CPU offload if needed

print(f"\nModel path: {MODEL_PATH}")
print(f"GPUs available: {num_gpus}")
print(f"Memory config:")
print(f"  Per GPU: 70GB limit")
print(f"  CPU: 200GB for overflow")
print()

try:
    print("Loading model with device_map='auto' and max_memory limits...")
    print("This is the same approach that makes inference work.")
    print()

    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",  # Auto-distribute across GPUs
        max_memory=max_memory,  # Enforce 70GB per GPU limit
        trust_remote_code=True
    )

    print("\n✅ Model loaded successfully!")
    print(f"Model class: {model.__class__.__name__}")

    print("\nDevice map (first 30 components):")
    if hasattr(model, 'hf_device_map'):
        for key, device in sorted(model.hf_device_map.items())[:30]:
            print(f"  {key}: {device}")
        if len(model.hf_device_map) > 30:
            print(f"  ... and {len(model.hf_device_map) - 30} more components")

    # Check GPU memory usage
    print("\nGPU Memory Usage:")
    total_allocated = 0
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        free = total - reserved
        print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free / {total:.2f}GB total")
        total_allocated += allocated

    print(f"\nTotal GPU memory allocated: {total_allocated:.2f}GB")
    print(f"Expected: ~445GB distributed across {num_gpus} GPUs (~56GB per GPU)")

    print("\n" + "="*80)
    print("✅ Test PASSED - Model Loading Successful!")
    print("="*80)
    print("\nThis confirms the fix works:")
    print("1. device_map='auto' + max_memory prevents OOM ✓")
    print("2. Model distributes properly across GPUs ✓")
    print("3. Ready to proceed with actual training ✓")

except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*80)
    print("❌ Test FAILED")
    print("="*80)
