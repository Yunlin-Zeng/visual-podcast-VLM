#!/bin/bash
set -e

# Configuration
BASE_MODEL_235B="/home/ubuntu/LLM/qwen3-vl-235b"
BASE_MODEL_32B="/home/ubuntu/LLM/qwen3-vl-32b"
LORA_PATH="/home/ubuntu/image-to-text/Qwen3-VL/finetuned_models/2025-11-28_qwen3vl-32b-sporc-v3"
SAMPLES_DIR="/home/ubuntu/image-to-text/Qwen3-VL/inference_samples_new30"
OUTPUT_DIR_235B="/home/ubuntu/LLM/qwen3-vl-235b/2025-12-04_evaluation_235b_new30"
OUTPUT_DIR_32B="/home/ubuntu/LLM/qwen3-vl-32b/2025-12-04_evaluation_32b_v3_new30"
PROMPT_FILE="/home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin/evaluation_prompt.txt"
INFERENCE_SCRIPT="/home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin/inference.py"
SEED=0
NUM_GPUS=8

cd /home/ubuntu/image-to-text/Qwen3-VL

echo "=========================================="
echo "Step 1: Running 235B on replacement sample_49"
echo "=========================================="

# Run 235B on just sample_49
python "$INFERENCE_SCRIPT" --mode single \
    --model-path "$BASE_MODEL_235B" \
    --sample-dir "$SAMPLES_DIR/sample_49_story_45530" \
    --output-dir "$OUTPUT_DIR_235B" \
    --prompt-file "$PROMPT_FILE" \
    --seed $SEED

echo ""
echo "=========================================="
echo "Step 2: Running 32B v3 on all 30 samples (8 parallel)"
echo "=========================================="

mkdir -p "$OUTPUT_DIR_32B"

# Get all sample directories
SAMPLE_DIRS=($(ls -d ${SAMPLES_DIR}/sample_* | sort -V))
echo "Found ${#SAMPLE_DIRS[@]} samples"

# Launch inference in batches of NUM_GPUS
for i in "${!SAMPLE_DIRS[@]}"; do
    SAMPLE_DIR="${SAMPLE_DIRS[$i]}"
    SAMPLE_NAME=$(basename "$SAMPLE_DIR")
    GPU_ID=$((i % NUM_GPUS))
    
    echo "[$((i+1))/${#SAMPLE_DIRS[@]}] Launching $SAMPLE_NAME on GPU $GPU_ID"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$INFERENCE_SCRIPT" --mode single \
        --model-path "$BASE_MODEL_32B" \
        --lora-adapter "$LORA_PATH" \
        --sample-dir "$SAMPLE_DIR" \
        --output-dir "$OUTPUT_DIR_32B" \
        --prompt-file "$PROMPT_FILE" \
        --seed $SEED \
        > "${OUTPUT_DIR_32B}/log_${SAMPLE_NAME}.txt" 2>&1 &
    
    # Wait after every NUM_GPUS launches
    if [ $(((i + 1) % NUM_GPUS)) -eq 0 ]; then
        echo "Waiting for batch to complete..."
        wait
        echo "Batch complete."
    fi
done

# Wait for any remaining
wait
echo ""
echo "=========================================="
echo "All inferences complete!"
echo "235B output: $OUTPUT_DIR_235B"
echo "32B v3 output: $OUTPUT_DIR_32B"
echo "=========================================="
