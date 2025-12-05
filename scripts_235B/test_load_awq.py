#!/usr/bin/env python3
"""
Test loading AWQ quantized Qwen3-VL-235B model.

AWQ (Activation-aware Weight Quantization):
- Keeps weights in int4 format during inference
- Should load directly without loading BF16 first
- Expected memory: ~15GB per GPU (117GB / 8 GPUs)
- Supports fine-tuning with PEFT/LoRA
"""

import torch
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

print("="*80)
print("Testing AWQ Quantized Model Loading - Attempt 13")
print("="*80)

MODEL_PATH = "/home/ubuntu/LLM/qwen3-vl-235b-awq"

print(f"\nModel path: {MODEL_PATH}")
print(f"Model: Qwen3-VL-235B-A22B-Instruct-AWQ")
print(f"Size: 117GB (AWQ int4 quantization)")
print(f"Expected per-GPU: ~15GB")
print()

try:
    print("Loading AWQ model...")
    print("  AWQ keeps weights quantized (int4) during inference")
    print("  Should NOT load BF16 first like bitsandbytes")
    print()

    # Load AWQ model - should handle quantization automatically
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
    )

    print("\n✅ Model loaded successfully!")
    print(f"Model class: {model.__class__.__name__}")

    # Check if model has quantization config
    if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
        print(f"Quantization config: {model.config.quantization_config}")

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
    print(f"Expected: ~117GB for full model in AWQ int4")

    # Try loading processor too
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("✅ Processor loaded successfully!")

    print("\n" + "="*80)
    print("✅ Test PASSED - AWQ Model Loaded Successfully!")
    print("="*80)
    print("\nNext steps:")
    print("1. Model fits in GPU memory ✓")
    print("2. Ready for LoRA fine-tuning with PEFT")
    print("3. Can proceed with training script")

except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*80)
    print("❌ Test FAILED")
    print("="*80)
