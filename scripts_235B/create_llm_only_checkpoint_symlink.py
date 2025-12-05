"""
Create a filtered checkpoint using SYMLINKS instead of copying files.

This approach:
1. Creates symbolic links to original shard files (no disk space needed)
2. Creates a new filtered weight index excluding vision weights
3. Copies only config files (~10MB total)

This saves ~439GB of disk space compared to full copy.
"""

import json
import shutil
import os
from pathlib import Path

# Paths
ORIGINAL_MODEL_PATH = Path("/home/ubuntu/LLM/qwen3-vl-235b")
OUTPUT_MODEL_PATH = Path("/home/ubuntu/LLM/qwen3-vl-235b-llm-only")

print("=" * 80)
print("Creating LLM-only checkpoint using SYMLINKS")
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

# 3. Find which shard files are needed
print("\nStep 3: Identifying required shard files...")
required_shards = set(llm_weight_map.values())
all_shards = set(original_weight_map.values())

print(f"  Total shards: {len(all_shards)}")
print(f"  Required for LLM: {len(required_shards)}")

# 4. Create symlinks to shard files (instead of copying)
print("\nStep 4: Creating symlinks to shard files...")
for shard_file in sorted(required_shards):
    src = ORIGINAL_MODEL_PATH / shard_file
    dst = OUTPUT_MODEL_PATH / shard_file

    if not dst.exists():
        os.symlink(src, dst)

print(f"  ✓ Created {len(required_shards)} symlinks (no disk space used!)")

# 5. Create new weight index
print("\nStep 5: Creating new weight index...")
new_index_data = {
    "metadata": {
        "note": "LLM-only checkpoint - vision encoder and MLP weights excluded",
        "original_path": str(ORIGINAL_MODEL_PATH)
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
print("LLM-only checkpoint created successfully using SYMLINKS!")
print("=" * 80)
print(f"Output directory: {OUTPUT_MODEL_PATH}")
print(f"\nWeight statistics:")
print(f"  Original model: {len(original_weight_map)} weights")
print(f"  LLM-only model: {len(llm_weight_map)} weights")
print(f"  Excluded: {len(vision_weights)} vision weights")
print(f"\nDisk usage:")
print(f"  Symlinks created: {len(required_shards)} (no disk space used)")
print(f"  Config files copied: ~10MB")
print(f"  Total disk space saved: ~439GB")
print(f"\nExpected memory savings during training: ~10-15GB per GPU")
print("=" * 80)
