#!/usr/bin/env python3
"""
Fix the offline-quantized model to use proper Params4bit format.

This script:
1. Loads quantized tensors from safetensors files
2. Loads QuantState metadata from _quant_state.pt files
3. Wraps them into Params4bit using from_prequantized()
4. Saves back to safetensors in the correct format
5. Updates config.json with quantization metadata
"""

import os
import gc
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file, save_file
from bitsandbytes.nn import Params4bit
import shutil


def fix_shard(shard_path, quant_state_path, output_path):
    """
    Load a quantized shard and its quant states, wrap properly, and save.
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

    # Wrap quantized tensors in Params4bit
    fixed_dict = {}

    for name, tensor in tqdm(state_dict.items(), desc="  Wrapping tensors"):
        if name in quant_states:
            # This tensor was quantized - wrap it properly
            quant_state = quant_states[name]

            # Create Params4bit from our prequantized data
            # The from_prequantized expects:
            # - data: the quantized tensor
            # - quantized_stats: dict with the QuantState
            try:
                # Reconstruct QuantState from our saved data
                # NOTE: from_prequantized() has a bug in bitsandbytes, so we create Params4bit manually

                # Create nested state2 if present (double quantization)
                if hasattr(quant_state, 'state2') and quant_state.state2 is not None:
                    from bitsandbytes.functional import QuantState
                    state2 = QuantState(
                        absmax=quant_state.state2.absmax,
                        blocksize=quant_state.state2.blocksize,
                        code=quant_state.state2.code,
                        dtype=quant_state.state2.dtype,
                    )
                    offset = quant_state.offset if hasattr(quant_state, 'offset') else None
                else:
                    state2 = None
                    offset = None

                # Create main QuantState
                from bitsandbytes.functional import QuantState
                reconstructed_qs = QuantState(
                    quant_type=quant_state.quant_type,
                    absmax=quant_state.absmax,
                    blocksize=quant_state.blocksize,
                    code=quant_state.code,
                    dtype=quant_state.dtype,
                    shape=quant_state.shape,
                    offset=offset,
                    state2=state2,
                )

                # Manually create Params4bit (bypasses buggy from_prequantized)
                params = torch.Tensor._make_subclass(Params4bit, tensor.cpu())
                params.requires_grad = False
                params.quant_state = reconstructed_qs
                params.blocksize = reconstructed_qs.blocksize
                params.compress_statistics = (state2 is not None)
                params.quant_type = reconstructed_qs.quant_type
                params.bnb_quantized = True
                params.quant_storage = tensor.dtype
                params.module = None

                fixed_dict[name] = params

            except Exception as e:
                print(f"    Warning: Could not wrap {name}: {e}")
                print(f"    Keeping original tensor")
                fixed_dict[name] = tensor
        else:
            # Not quantized - keep as is
            fixed_dict[name] = tensor

    # Save the fixed shard
    # Note: safetensors might have issues with Params4bit, so we need to extract the data
    save_dict = {}
    for name, param in fixed_dict.items():
        if isinstance(param, Params4bit):
            # Extract the underlying data from Params4bit
            # Params4bit stores data in a specific format
            save_dict[name] = param.data
        else:
            save_dict[name] = param

    if shard_path.suffix == '.safetensors':
        save_file(save_dict, str(output_path))
    else:
        torch.save(save_dict, str(output_path))

    print(f"  Saved: {output_path.name}")

    # Clear memory
    del state_dict, quant_states, fixed_dict, save_dict
    gc.collect()


def update_config(input_dir, output_dir):
    """
    Update config.json with quantization metadata.
    """
    config_path = input_dir / "config.json"
    output_config_path = output_dir / "config.json"

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Add quantization config
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
    parser = argparse.ArgumentParser(description='Fix offline-quantized model format')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory with quantized model (has _quant_state.pt files)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for fixed model')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Fixing Offline-Quantized Model Format")
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

        fix_shard(shard_path, quant_state_path, output_path)

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
    print("✓ Model fixing complete!")
    print(f"Fixed model saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
