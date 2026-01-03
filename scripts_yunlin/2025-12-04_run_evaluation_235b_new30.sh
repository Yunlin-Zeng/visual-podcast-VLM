#!/bin/bash
# 235B base model evaluation on NEW 30 samples (21-50)
# Uses all 8 GPUs for model distribution

set -e

# Configuration
BASE_MODEL="/home/ubuntu/LLM/qwen3-vl-235b"
ALL_SAMPLES_DIR="/home/ubuntu/image-to-text/Qwen3-VL/inference_samples"
NEW_SAMPLES_DIR="/home/ubuntu/image-to-text/Qwen3-VL/inference_samples_new30"
PROMPT_FILE="/home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin/evaluation_prompt.txt"
SEED=0
DATE=$(date +%Y-%m-%d)

# Output directory
OUTPUT_DIR="/home/ubuntu/LLM/qwen3-vl-235b/${DATE}_evaluation_235b_new30"
mkdir -p "$OUTPUT_DIR"

# Create temporary directory with symlinks to only new samples (21-50)
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

echo ""
echo "================================================================================"
echo "235B Base Model Evaluation - NEW 30 Samples (21-50)"
echo "================================================================================"
echo "Model: $BASE_MODEL"
echo "Samples: $NEW_SAMPLES_DIR ($(ls -1 $NEW_SAMPLES_DIR | wc -l) samples)"
echo "Output: $OUTPUT_DIR"
echo "================================================================================"
echo ""
echo "Estimated time: 30 samples × 8 min/sample = ~4 hours"
echo ""

cd /home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin

# Run batch evaluation
python inference.py --mode batch \
    --model-path "$BASE_MODEL" \
    --samples-dir "$NEW_SAMPLES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --seed $SEED

echo ""
echo "================================================================================"
echo "Evaluation complete!"
echo "Output directory: $OUTPUT_DIR"
echo "================================================================================"
