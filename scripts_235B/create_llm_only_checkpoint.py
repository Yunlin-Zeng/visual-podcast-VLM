"""
Create a filtered checkpoint that excludes vision encoder and MLP weights.

This script:
1. Reads the original 235B checkpoint index
2. Filters out all vision-related weights (model.visual.*)
3. Creates a new checkpoint directory with only LLM weights
4. Copies config files and updates the weight index

Expected memory savings: ~10-15GB per GPU (351 vision weights excluded)
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
ORIGINAL_MODEL_PATH = Path("/home/ubuntu/LLM/qwen3-vl-235b")
OUTPUT_MODEL_PATH = Path("/home/ubuntu/LLM/qwen3-vl-235b-llm-only")

print("=" * 80)
print("Creating LLM-only checkpoint (excluding vision encoder + MLP)")
print("=" * 80)
print(f"Source: {ORIGINAL_MODEL_PATH}")
print(f"Output: {OUTPUT_MODEL_PATH}")
print()

# Create output directory
OUTPUT_MODEL_PATH.mkdir(exist_ok=True)

# 1. Load original weight index
print("Step 1: Loading original weight index...")
index_file = ORIGINAL_MODEL_PATH / "model.safetensors.index.json"
with open(index_file) as f:
    index_data = json.load(f)

original_weight_map = index_data["weight_map"]
print(f"  Total weights in original model: {len(original_weight_map)}")

# 2. Filter out vision weights
print("\nStep 2: Filtering vision weights...")
llm_weight_map = {}
vision_weights = []

for weight_name, shard_file in original_weight_map.items():
    if "visual" in weight_name:
        vision_weights.append(weight_name)
    else:
        llm_weight_map[weight_name] = shard_file

print(f"  LLM weights: {len(llm_weight_map)}")
print(f"  Vision weights (excluded): {len(vision_weights)}")
print(f"\nExample vision weights excluded:")
for w in vision_weights[:5]:
    print(f"    - {w}")

# 3. Find which shard files are needed for LLM weights
print("\nStep 3: Identifying required shard files...")
required_shards = set(llm_weight_map.values())
all_shards = set(original_weight_map.values())
excluded_shards = all_shards - required_shards

print(f"  Total shards: {len(all_shards)}")
print(f"  Required for LLM: {len(required_shards)}")
print(f"  Purely vision (can skip): {len(excluded_shards)}")

# 4. Copy required shard files
print("\nStep 4: Copying required shard files...")
print("  (This may take a while - 235B model is large)")

for shard_file in tqdm(sorted(required_shards), desc="Copying shards"):
    src = ORIGINAL_MODEL_PATH / shard_file
    dst = OUTPUT_MODEL_PATH / shard_file

    if not dst.exists():
        shutil.copy2(src, dst)

print(f"  ✓ Copied {len(required_shards)} shard files")

# 5. Create new weight index
print("\nStep 5: Creating new weight index...")
new_index_data = {
    "metadata": {
        "total_size": sum(
            (ORIGINAL_MODEL_PATH / f).stat().st_size
            for f in required_shards
        ),
        "note": "LLM-only checkpoint - vision encoder and MLP weights excluded"
    },
    "weight_map": llm_weight_map
}

new_index_file = OUTPUT_MODEL_PATH / "model.safetensors.index.json"
with open(new_index_file, 'w') as f:
    json.dump(new_index_data, f, indent=2)

print(f"  ✓ Created {new_index_file}")

# 6. Copy config files
print("\nStep 6: Copying config files...")
config_files = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
    "preprocessor_config.json",
]

for config_file in config_files:
    src = ORIGINAL_MODEL_PATH / config_file
    if src.exists():
        dst = OUTPUT_MODEL_PATH / config_file
        shutil.copy2(src, dst)
        print(f"  ✓ Copied {config_file}")

# 7. Summary
print("\n" + "=" * 80)
print("LLM-only checkpoint created successfully!")
print("=" * 80)
print(f"Output directory: {OUTPUT_MODEL_PATH}")
print(f"\nWeight statistics:")
print(f"  Original model: {len(original_weight_map)} weights")
print(f"  LLM-only model: {len(llm_weight_map)} weights")
print(f"  Excluded: {len(vision_weights)} vision weights")
print(f"\nShard statistics:")
print(f"  Original: {len(all_shards)} shard files")
print(f"  LLM-only: {len(required_shards)} shard files")
print(f"  Saved: {len(excluded_shards)} purely-vision shard files (not copied)")
print(f"\nExpected memory savings during training: ~10-15GB per GPU")
print("=" * 80)
