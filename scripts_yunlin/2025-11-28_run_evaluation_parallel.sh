#!/bin/bash
# Parallel evaluation script - runs 8 samples concurrently on 8 GPUs
# Evaluates both checkpoint-126 and checkpoint-62

set -e

# Configuration
BASE_MODEL="/home/ubuntu/LLM/qwen3-vl-32b"
LORA_DIR="/home/ubuntu/image-to-text/Qwen3-VL/finetuned_models/2025-11-28_qwen3vl-32b-sporc-v3"
SAMPLES_DIR="/home/ubuntu/image-to-text/Qwen3-VL/inference_samples"
PROMPT_FILE="/home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin/evaluation_prompt.txt"
SEED=0
DATE=$(date +%Y-%m-%d)
NUM_GPUS=8

# Checkpoints to evaluate
CHECKPOINTS=("checkpoint-126" "checkpoint-62")

cd /home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin

# Get all sample directories
SAMPLE_DIRS=($(ls -d ${SAMPLES_DIR}/sample_* | sort))
NUM_SAMPLES=${#SAMPLE_DIRS[@]}

echo "================================================================================"
echo "Parallel Evaluation (8 GPUs)"
echo "================================================================================"
echo "Base model: $BASE_MODEL"
echo "LoRA dir: $LORA_DIR"
echo "Checkpoints: ${CHECKPOINTS[*]}"
echo "Samples: $NUM_SAMPLES"
echo "================================================================================"

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    echo ""
    echo "================================================================================"
    echo "Evaluating: $CHECKPOINT"
    echo "================================================================================"

    # Output directory
    OUTPUT_DIR="${LORA_DIR}/${CHECKPOINT}/${DATE}_evaluation"
    mkdir -p "$OUTPUT_DIR"

    # Save prompt to output dir
    cp "$PROMPT_FILE" "$OUTPUT_DIR/input_prompt.txt"

    echo "Output: $OUTPUT_DIR"
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
            --lora-adapter "${LORA_DIR}/${CHECKPOINT}" \
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
    echo "$CHECKPOINT complete: $COMPLETED/$NUM_SAMPLES samples"
done

echo ""
echo "================================================================================"
echo "All evaluations complete!"
echo "================================================================================"
