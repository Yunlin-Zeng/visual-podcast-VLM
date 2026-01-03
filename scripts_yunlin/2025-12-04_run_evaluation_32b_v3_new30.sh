#!/bin/bash
# Parallel evaluation script - runs 8 samples concurrently on 8 GPUs
# 32B v3 finetuned model on NEW 30 samples (21-50)

set -e

# Configuration
BASE_MODEL="/home/ubuntu/LLM/qwen3-vl-32b"
LORA_PATH="/home/ubuntu/image-to-text/Qwen3-VL/finetuned_models/2025-11-28_qwen3vl-32b-sporc-v3"
ALL_SAMPLES_DIR="/home/ubuntu/image-to-text/Qwen3-VL/inference_samples"
NEW_SAMPLES_DIR="/home/ubuntu/image-to-text/Qwen3-VL/inference_samples_new30"
PROMPT_FILE="/home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin/evaluation_prompt.txt"
SEED=0
DATE=$(date +%Y-%m-%d)
NUM_GPUS=8

# Output directory
OUTPUT_DIR="${LORA_PATH}/${DATE}_evaluation_new30"
mkdir -p "$OUTPUT_DIR"

# Create temporary directory with symlinks to only new samples (21-50) if not exists
if [ ! -d "$NEW_SAMPLES_DIR" ] || [ $(ls -1 "$NEW_SAMPLES_DIR" 2>/dev/null | wc -l) -eq 0 ]; then
    echo "Creating symlinks for samples 21-50..."
    mkdir -p "$NEW_SAMPLES_DIR"
    for i in $(seq 21 50); do
        sample_num=$(printf "%02d" $i)
        sample_dir=$(ls -d ${ALL_SAMPLES_DIR}/sample_${sample_num}_* 2>/dev/null | head -1)
        if [ -n "$sample_dir" ] && [ -d "$sample_dir" ]; then
            link_name="${NEW_SAMPLES_DIR}/$(basename $sample_dir)"
            if [ ! -L "$link_name" ]; then
                ln -s "$sample_dir" "$link_name"
                echo "  Linked: $(basename $sample_dir)"
            fi
        fi
    done
fi

cd /home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin

# Get all sample directories
SAMPLE_DIRS=($(ls -d ${NEW_SAMPLES_DIR}/sample_* | sort))
NUM_SAMPLES=${#SAMPLE_DIRS[@]}

echo ""
echo "================================================================================"
echo "32B v3 Parallel Evaluation (8 GPUs) - NEW 30 Samples"
echo "================================================================================"
echo "Base model: $BASE_MODEL"
echo "LoRA path: $LORA_PATH"
echo "Samples: $NUM_SAMPLES"
echo "Output: $OUTPUT_DIR"
echo "================================================================================"
echo ""
echo "Estimated time: 30 samples / 8 parallel × 4.5 min = ~17 minutes"
echo ""

# Save prompt to output dir
cp "$PROMPT_FILE" "$OUTPUT_DIR/input_prompt.txt"

echo "Launching $NUM_GPUS parallel processes..."
echo ""

# Launch inference for each sample, round-robin across GPUs
for i in "${!SAMPLE_DIRS[@]}"; do
    SAMPLE_DIR="${SAMPLE_DIRS[$i]}"
    SAMPLE_NAME=$(basename "$SAMPLE_DIR")
    GPU_ID=$((i % NUM_GPUS))

    # Check if output already exists (skip if done)
    OUTPUT_FILE="${OUTPUT_DIR}/${SAMPLE_NAME}.txt"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "[$((i+1))/$NUM_SAMPLES] $SAMPLE_NAME - already done, skipping"
        continue
    fi

    echo "[$((i+1))/$NUM_SAMPLES] $SAMPLE_NAME -> GPU $GPU_ID"

    CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py --mode single \
        --model-path "$BASE_MODEL" \
        --lora-adapter "$LORA_PATH" \
        --sample-dir "$SAMPLE_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --prompt-file "$PROMPT_FILE" \
        --seed $SEED \
        > "${OUTPUT_DIR}/log_${SAMPLE_NAME}.txt" 2>&1 &

    # Wait if we've launched NUM_GPUS processes
    if [ $(((i + 1) % NUM_GPUS)) -eq 0 ]; then
        echo "Waiting for batch to complete..."
        wait
        echo "Batch complete!"
        echo ""
    fi
done

# Wait for any remaining processes
echo "Waiting for final batch..."
wait

# Count completed samples
COMPLETED=$(ls ${OUTPUT_DIR}/sample_*.txt 2>/dev/null | wc -l)

echo ""
echo "================================================================================"
echo "Evaluation complete!"
echo "================================================================================"
echo "Completed: $COMPLETED/$NUM_SAMPLES samples"
echo "Output: $OUTPUT_DIR"
echo "================================================================================"
