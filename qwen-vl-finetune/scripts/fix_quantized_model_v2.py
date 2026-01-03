#!/usr/bin/env python3
"""
Fix the offline-quantized model to use proper bitsandbytes serialization format.

The issue with v1: safetensors can't store Params4bit objects.
The solution in v2: Store QuantState info in safetensors metadata, keep tensors in original BF16 format.

Actually, after investigation, I realize the real issue:
- Transformers doesn't support loading pre-quantized models from disk
- The quantization_config in config.json tells transformers to quantize DURING loading
- We need to keep the model in BF16 format and let transformers quantize it on-the-fly

So this script will DEQUANTIZE the model back to BF16.
"""

import os
import gc
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file, save_file
import shutil


def dequantize_shard(shard_path, quant_state_path, output_path):
    """
    Load a quantized shard and its quant states, dequantize back to BF16, and save.
    """
    print(f"\nProcessing: {shard_path.name}")

    # Load quantized tensors
    if shard_path.suffix == '.safetensors':
        state_dict = load_file(str(shard_path))
    else:
        state_dict = torch.load(str(shard_path), map_location='cpu')

    # Load quantization states
    if quant_state_path.exists():
        quant_states = torch.load(str(quant_state_path), map_location='cpu', weights_only=False)
        print(f"  Loaded {len(quant_states)} quant states")
    else:
        print(f"  Warning: No quant_state file found at {quant_state_path}")
        quant_states = {}

    # Dequantize tensors back to BF16
    dequantized_dict = {}

    for name, tensor in tqdm(state_dict.items(), desc="  Dequantizing tensors"):
        if name in quant_states:
            # This tensor was quantized - dequantize it
            quant_state = quant_states[name]

            try:
                from bitsandbytes.functional import dequantize_4bit

                # Dequantize back to original dtype
                dequantized = dequantize_4bit(
                    tensor,
                    quant_state,
                    quant_type=quant_state.quant_type
                )

                # Reshape to original shape
                dequantized = dequantized.reshape(quant_state.shape).to(quant_state.dtype)

                dequantized_dict[name] = dequantized

            except Exception as e:
                print(f"    Warning: Could not dequantize {name}: {e}")
                print(f"    Keeping original tensor")
                dequantized_dict[name] = tensor
        else:
            # Not quantized - keep as is
            dequantized_dict[name] = tensor

    # Save the dequantized shard
    if shard_path.suffix == '.safetensors':
        save_file(dequantized_dict, str(output_path))
    else:
        torch.save(dequantized_dict, str(output_path))

    print(f"  Saved: {output_path.name}")

    # Clear memory
    del state_dict, quant_states, dequantized_dict
    gc.collect()


def update_config(input_dir, output_dir):
    """
    Copy config.json and ADD quantization metadata (for on-the-fly quantization).
    """
    config_path = input_dir / "config.json"
    output_config_path = output_dir / "config.json"

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Add quantization config for on-the-fly quantization
        config['quantization_config'] = {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }

        with open(output_config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Updated config.json with quantization metadata")
    else:
        print(f"Warning: config.json not found at {config_path}")


def main():
    parser = argparse.ArgumentParser(description='Dequantize offline-quantized model back to BF16')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory with quantized model (has _quant_state.pt files)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for BF16 model')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Dequantizing Model Back to BF16")
    print("="*80)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    # Find all shard files
    shard_files = sorted(input_dir.glob("model-*.safetensors"))
    if not shard_files:
        shard_files = sorted(input_dir.glob("model-*.bin"))

    print(f"Found {len(shard_files)} model shards")

    # Process each shard
    for shard_path in shard_files:
        quant_state_path = input_dir / f"{shard_path.stem}_quant_state.pt"
        output_path = output_dir / shard_path.name

        dequantize_shard(shard_path, quant_state_path, output_path)

    # Copy other necessary files
    print("\nCopying other model files...")
    for file_name in ['config.json', 'generation_config.json', 'tokenizer_config.json',
                      'tokenizer.json', 'vocab.json', 'merges.txt',
                      'preprocessor_config.json', 'model.safetensors.index.json']:
        src = input_dir / file_name
        dst = output_dir / file_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ Copied: {file_name}")

    # Update config with quantization metadata
    update_config(input_dir, output_dir)

    print("\n" + "="*80)
    print("✓ Dequantization complete!")
    print(f"BF16 model saved to: {output_dir}")
    print("Note: This model will be quantized on-the-fly during loading with load_in_4bit=True")
    print("="*80)


if __name__ == "__main__":
    main()
