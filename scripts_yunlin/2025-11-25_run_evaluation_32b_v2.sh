#!/bin/bash
# Run evaluation on 32B model v2 checkpoints (skip base - already evaluated)
# Usage: bash 2025-11-25_run_evaluation_32b_v2.sh

set -e

# Configuration
BASE_MODEL="/home/ubuntu/LLM/qwen3-vl-32b"
LORA_DIR="/home/ubuntu/image-to-text/Qwen3-VL/finetuned_models/2025-11-25_qwen3vl-32b-sporc-v2"
SAMPLES_DIR="/home/ubuntu/image-to-text/Qwen3-VL/inference_samples"
PROMPT_FILE="/home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin/evaluation_prompt.txt"
SEED=0
DATE=$(date +%Y-%m-%d)

cd /home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin

echo "================================================================================"
echo "32B Model v2 Evaluation (4 checkpoints, seed=$SEED)"
echo "================================================================================"
echo "Base model: $BASE_MODEL"
echo "LoRA dir: $LORA_DIR"
echo "Samples: $SAMPLES_DIR"
echo "Seed: $SEED"
echo "================================================================================"

# 1. Evaluate checkpoint-14
echo ""
echo "================================================================================"
echo "[1/4] Evaluating: checkpoint-14"
echo "================================================================================"
CHECKPOINT="${LORA_DIR}/checkpoint-14"
OUTPUT_DIR="${CHECKPOINT}/${DATE}_evaluation"
python inference.py --mode single \
    --model-path "$BASE_MODEL" \
    --lora-adapter "$CHECKPOINT" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --seed $SEED
echo "checkpoint-14 evaluation complete: $OUTPUT_DIR"

# 2. Evaluate checkpoint-28
echo ""
echo "================================================================================"
echo "[2/4] Evaluating: checkpoint-28"
echo "================================================================================"
CHECKPOINT="${LORA_DIR}/checkpoint-28"
OUTPUT_DIR="${CHECKPOINT}/${DATE}_evaluation"
python inference.py --mode single \
    --model-path "$BASE_MODEL" \
    --lora-adapter "$CHECKPOINT" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --seed $SEED
echo "checkpoint-28 evaluation complete: $OUTPUT_DIR"

# 3. Evaluate checkpoint-42
echo ""
echo "================================================================================"
echo "[3/4] Evaluating: checkpoint-42"
echo "================================================================================"
CHECKPOINT="${LORA_DIR}/checkpoint-42"
OUTPUT_DIR="${CHECKPOINT}/${DATE}_evaluation"
python inference.py --mode single \
    --model-path "$BASE_MODEL" \
    --lora-adapter "$CHECKPOINT" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --seed $SEED
echo "checkpoint-42 evaluation complete: $OUTPUT_DIR"

# 4. Evaluate checkpoint-56
echo ""
echo "================================================================================"
echo "[4/4] Evaluating: checkpoint-56"
echo "================================================================================"
CHECKPOINT="${LORA_DIR}/checkpoint-56"
OUTPUT_DIR="${CHECKPOINT}/${DATE}_evaluation"
python inference.py --mode single \
    --model-path "$BASE_MODEL" \
    --lora-adapter "$CHECKPOINT" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --seed $SEED
echo "checkpoint-56 evaluation complete: $OUTPUT_DIR"

echo ""
echo "================================================================================"
echo "Evaluation complete!"
echo "================================================================================"
echo "Results saved to:"
echo "  - CP-14: ${LORA_DIR}/checkpoint-14/${DATE}_evaluation/"
echo "  - CP-28: ${LORA_DIR}/checkpoint-28/${DATE}_evaluation/"
echo "  - CP-42: ${LORA_DIR}/checkpoint-42/${DATE}_evaluation/"
echo "  - CP-56: ${LORA_DIR}/checkpoint-56/${DATE}_evaluation/"
