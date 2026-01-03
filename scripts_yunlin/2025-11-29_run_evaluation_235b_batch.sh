#!/bin/bash
# 235B base model evaluation using batch mode (loads model once, runs all samples)
# Uses all 8 GPUs for model distribution

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

echo "================================================================================"
echo "235B Base Model Evaluation - Batch Mode (loads model once)"
echo "================================================================================"
echo "Model: $BASE_MODEL"
echo "Samples: $SAMPLES_DIR"
echo "Output: $OUTPUT_DIR"
echo "================================================================================"
echo ""

# Run batch evaluation (loads model once, processes all samples sequentially)
python inference.py --mode batch \
    --model-path "$BASE_MODEL" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --seed $SEED

echo ""
echo "================================================================================"
echo "Evaluation complete!"
echo "Output directory: $OUTPUT_DIR"
echo "================================================================================"
