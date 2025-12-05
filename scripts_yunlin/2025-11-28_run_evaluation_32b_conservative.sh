#!/bin/bash
# Run evaluation on 32B model trained with conservative settings (4004 samples, 1 epoch)
# Checkpoints: 126 (final), 62 (mid)

set -e

# Configuration
BASE_MODEL="/home/ubuntu/LLM/qwen3-vl-32b"
LORA_DIR="/home/ubuntu/image-to-text/Qwen3-VL/finetuned_models/2025-11-27_qwen3vl-32b-sporc-4004-conservative"
SAMPLES_DIR="/home/ubuntu/image-to-text/Qwen3-VL/inference_samples"
PROMPT_FILE="/home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin/evaluation_prompt.txt"
SEED=0
DATE=$(date +%Y-%m-%d)

cd /home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin

echo "================================================================================"
echo "32B Model Evaluation (conservative: LR=4e-6, NEFTune=5.0, 1 epoch)"
echo "================================================================================"
echo "Base model: $BASE_MODEL"
echo "LoRA dir: $LORA_DIR"
echo "Samples: $SAMPLES_DIR"
echo "Seed: $SEED"
echo "================================================================================"

# 1. Evaluate checkpoint-126 (final)
echo ""
echo "================================================================================"
echo "[1/2] Evaluating: checkpoint-126 (final)"
echo "================================================================================"
CHECKPOINT="${LORA_DIR}/checkpoint-126"
OUTPUT_DIR="${CHECKPOINT}/${DATE}_evaluation"
python inference.py --mode single \
    --model-path "$BASE_MODEL" \
    --lora-adapter "$CHECKPOINT" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --seed $SEED
echo "checkpoint-126 evaluation complete: $OUTPUT_DIR"

# 2. Evaluate checkpoint-62 (mid epoch)
echo ""
echo "================================================================================"
echo "[2/2] Evaluating: checkpoint-62 (mid epoch)"
echo "================================================================================"
CHECKPOINT="${LORA_DIR}/checkpoint-62"
OUTPUT_DIR="${CHECKPOINT}/${DATE}_evaluation"
python inference.py --mode single \
    --model-path "$BASE_MODEL" \
    --lora-adapter "$CHECKPOINT" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --seed $SEED
echo "checkpoint-62 evaluation complete: $OUTPUT_DIR"

echo ""
echo "================================================================================"
echo "Evaluation complete!"
echo "================================================================================"
echo "Results saved to:"
echo "  - CP-126: ${LORA_DIR}/checkpoint-126/${DATE}_evaluation/"
echo "  - CP-62:  ${LORA_DIR}/checkpoint-62/${DATE}_evaluation/"
