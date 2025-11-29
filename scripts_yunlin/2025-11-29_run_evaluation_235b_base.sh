#!/bin/bash
# Sequential evaluation script for 235B base model
# Each inference uses all 8 GPUs, so we run samples one at a time

set -e

# Configuration
BASE_MODEL="/home/ubuntu/LLM/qwen3-vl-235b"
SAMPLES_DIR="/home/ubuntu/image-to-text/Qwen3-VL/inference_samples"
PROMPT_FILE="/home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin/evaluation_prompt.txt"
SEED=0
DATE=$(date +%Y-%m-%d)

# Output directory
OUTPUT_DIR="/home/ubuntu/LLM/qwen3-vl-235b/${DATE}_evaluation_235b_base"
mkdir -p "$OUTPUT_DIR"

cd /home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin

# Get all sample directories
SAMPLE_DIRS=($(ls -d ${SAMPLES_DIR}/sample_* | sort))
NUM_SAMPLES=${#SAMPLE_DIRS[@]}

echo "================================================================================"
echo "Sequential Evaluation - 235B Base Model (8 GPUs per inference)"
echo "================================================================================"
echo "Model: $BASE_MODEL"
echo "Output: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES"
echo "================================================================================"
echo ""

# Save prompt to output dir
cp "$PROMPT_FILE" "$OUTPUT_DIR/input_prompt.txt"

# Run each sample sequentially (235B needs all 8 GPUs)
for i in "${!SAMPLE_DIRS[@]}"; do
    SAMPLE_DIR="${SAMPLE_DIRS[$i]}"
    SAMPLE_NAME=$(basename "$SAMPLE_DIR")

    # Check if output already exists (skip if done)
    OUTPUT_FILE="${OUTPUT_DIR}/${SAMPLE_NAME}.txt"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "[$((i+1))/$NUM_SAMPLES] $SAMPLE_NAME - already done, skipping"
        continue
    fi

    echo "[$((i+1))/$NUM_SAMPLES] Processing $SAMPLE_NAME..."
    START_TIME=$(date +%s)

    # Run inference using all GPUs (no CUDA_VISIBLE_DEVICES restriction)
    python inference.py --mode single \
        --model-path "$BASE_MODEL" \
        --sample-dir "$SAMPLE_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --prompt-file "$PROMPT_FILE" \
        --seed $SEED \
        2>&1 | tee "${OUTPUT_DIR}/log_${SAMPLE_NAME}.txt"

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ -f "$OUTPUT_FILE" ]; then
        WORD_COUNT=$(wc -w < "$OUTPUT_FILE" | tr -d ' ')
        echo "[$((i+1))/$NUM_SAMPLES] $SAMPLE_NAME complete: ${WORD_COUNT} words, ${DURATION}s"
    else
        echo "[$((i+1))/$NUM_SAMPLES] $SAMPLE_NAME FAILED"
    fi
    echo ""
done

# Count completed samples
COMPLETED=$(ls ${OUTPUT_DIR}/sample_*.txt 2>/dev/null | wc -l)
echo "================================================================================"
echo "Evaluation complete: $COMPLETED/$NUM_SAMPLES samples"
echo "Output directory: $OUTPUT_DIR"
echo "================================================================================"
