"""
Check LoRA weights initialization after training script saves them

This script verifies that:
1. LoRA weights exist in the saved checkpoint
2. Shows the statistics of lora_A and lora_B weights
3. Checks if they're close to zero (freshly initialized)

Usage:
    python check_lora_init.py --checkpoint ./finetuned_models/qwen3vl-235b-lora-short-prompt
"""

import argparse
import torch
from pathlib import Path
from safetensors import safe_open


def check_lora_weights(checkpoint_path):
    """
    Check LoRA weight initialization in a saved checkpoint

    Args:
        checkpoint_path: Path to LoRA checkpoint directory
    """

    checkpoint_path = Path(checkpoint_path)

    print("=" * 80)
    print("LORA WEIGHT INITIALIZATION CHECK")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # Find adapter model file
    adapter_file = checkpoint_path / "adapter_model.safetensors"

    if not adapter_file.exists():
        print(f"❌ Error: adapter_model.safetensors not found in {checkpoint_path}")
        print(f"   Available files:")
        for f in checkpoint_path.iterdir():
            print(f"     - {f.name}")
        return

    print(f"✓ Found: {adapter_file}")
    print()

    # Load and analyze weights
    print("=" * 80)
    print("WEIGHT STATISTICS")
    print("=" * 80)

    lora_a_weights = []
    lora_b_weights = []
    other_weights = []

    with safe_open(adapter_file, framework="pt", device="cpu") as f:
        keys = list(f.keys())

        print(f"\nTotal parameters: {len(keys)}")
        print()

        # Categorize weights
        for key in keys:
            if 'lora_A' in key:
                lora_a_weights.append(key)
            elif 'lora_B' in key:
                lora_b_weights.append(key)
            else:
                other_weights.append(key)

        print(f"LoRA A weights: {len(lora_a_weights)}")
        print(f"LoRA B weights: {len(lora_b_weights)}")
        print(f"Other weights (e.g., visual.merger): {len(other_weights)}")
        print()

        # Analyze LoRA A weights (should be random small values)
        print("=" * 80)
        print("LoRA_A WEIGHTS (should be random small values, ~N(0, 0.01))")
        print("=" * 80)

        if lora_a_weights:
            # Sample first 3 lora_A weights
            for i, key in enumerate(lora_a_weights[:3]):
                tensor = f.get_tensor(key)
                print(f"\n{i+1}. {key}")
                print(f"   Shape: {tensor.shape}")
                print(f"   Mean: {tensor.float().mean().item():.8f}")
                print(f"   Std:  {tensor.float().std().item():.8f}")
                print(f"   Min:  {tensor.float().min().item():.8f}")
                print(f"   Max:  {tensor.float().max().item():.8f}")
                print(f"   Non-zero ratio: {(tensor != 0).float().mean().item():.6f}")

        # Analyze LoRA B weights (should be initialized to zero)
        print("\n" + "=" * 80)
        print("LoRA_B WEIGHTS (should be ZERO initially)")
        print("=" * 80)

        all_b_zero = True

        if lora_b_weights:
            # Sample first 3 lora_B weights
            for i, key in enumerate(lora_b_weights[:3]):
                tensor = f.get_tensor(key)
                is_zero = torch.allclose(tensor.float(), torch.zeros_like(tensor).float(), atol=1e-6)

                print(f"\n{i+1}. {key}")
                print(f"   Shape: {tensor.shape}")
                print(f"   Mean: {tensor.float().mean().item():.8f}")
                print(f"   Std:  {tensor.float().std().item():.8f}")
                print(f"   Min:  {tensor.float().min().item():.8f}")
                print(f"   Max:  {tensor.float().max().item():.8f}")
                print(f"   All zeros? {'✓ YES' if is_zero else '✗ NO'}")

                if not is_zero:
                    all_b_zero = False

        # Check all lora_B weights
        print("\n" + "=" * 80)
        print("CHECKING ALL LoRA_B WEIGHTS")
        print("=" * 80)

        non_zero_b_count = 0
        for key in lora_b_weights:
            tensor = f.get_tensor(key)
            if not torch.allclose(tensor.float(), torch.zeros_like(tensor).float(), atol=1e-6):
                non_zero_b_count += 1
                print(f"  ⚠ Non-zero LoRA_B found: {key}")
                print(f"     Mean: {tensor.float().mean().item():.8f}, Max: {tensor.float().max().item():.8f}")

        if non_zero_b_count == 0:
            print(f"  ✓ All {len(lora_b_weights)} LoRA_B weights are zero (correct initialization)")
        else:
            print(f"  ✗ {non_zero_b_count}/{len(lora_b_weights)} LoRA_B weights are non-zero!")

        # Analyze other weights (visual.merger if tune_mm_mlp=True)
        if other_weights:
            print("\n" + "=" * 80)
            print("OTHER TRAINABLE WEIGHTS (e.g., visual.merger)")
            print("=" * 80)
            print(f"\nFound {len(other_weights)} non-LoRA parameters")
            print("Sample:")
            for i, key in enumerate(other_weights[:5]):
                tensor = f.get_tensor(key)
                print(f"  {i+1}. {key}")
                print(f"     Shape: {tensor.shape}, Mean: {tensor.float().mean().item():.6f}")

    # Final verdict
    print("\n" + "=" * 80)
    print("INITIALIZATION CHECK RESULT")
    print("=" * 80)

    if all_b_zero and lora_a_weights and lora_b_weights:
        print("✓ PASS: LoRA weights are correctly initialized")
        print("  - LoRA_A: Random small values (Kaiming init)")
        print("  - LoRA_B: All zeros")
        print("  - Effective ΔW = LoRA_B @ LoRA_A = 0 @ random = 0")
        print("\nExpected inference behavior:")
        print("  Base model + this LoRA adapter = Base model (identical output)")
    else:
        print("✗ FAIL: LoRA initialization may be incorrect")
        if not all_b_zero:
            print("  - LoRA_B weights are not all zero!")
            print("  - Model may have been partially trained")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Check LoRA weight initialization in saved checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory"
    )

    args = parser.parse_args()

    check_lora_weights(args.checkpoint)


if __name__ == "__main__":
    main()
