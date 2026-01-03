#!/bin/bash
# Run evaluation on 32B model (base + all checkpoints)
# Usage: bash run_evaluation_32b.sh

set -e

# Configuration
BASE_MODEL="/home/ubuntu/LLM/qwen3-vl-32b"
LORA_DIR="/home/ubuntu/image-to-text/Qwen3-VL/finetuned_models/2025-11-22_qwen3vl-32b-sporc-lora"
SAMPLES_DIR="/home/ubuntu/image-to-text/Qwen3-VL/inference_samples"
PROMPT_FILE="/home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin/evaluation_prompt.txt"
SEED=42
DATE=$(date +%Y-%m-%d)

cd /home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin

echo "================================================================================"
echo "32B Model Evaluation"
echo "================================================================================"
echo "Base model: $BASE_MODEL"
echo "LoRA dir: $LORA_DIR"
echo "Samples: $SAMPLES_DIR"
echo "Seed: $SEED"
echo "================================================================================"

# 1. Evaluate base model (no LoRA)
echo ""
echo "================================================================================"
echo "[1/4] Evaluating: Base model (no fine-tuning)"
echo "================================================================================"
OUTPUT_DIR="${BASE_MODEL}/${DATE}_evaluation_32b_base"
python inference.py --mode single \
    --model-path "$BASE_MODEL" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --seed $SEED
echo "✓ Base model evaluation complete: $OUTPUT_DIR"

# 2. Evaluate checkpoint-112-backup
echo ""
echo "================================================================================"
echo "[2/4] Evaluating: checkpoint-112-backup"
echo "================================================================================"
CHECKPOINT="${LORA_DIR}/checkpoint-112-backup"
OUTPUT_DIR="${CHECKPOINT}/${DATE}_evaluation"
python inference.py --mode single \
    --model-path "$BASE_MODEL" \
    --lora-adapter "$CHECKPOINT" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --seed $SEED
echo "✓ checkpoint-112-backup evaluation complete: $OUTPUT_DIR"

# 3. Evaluate checkpoint-224
echo ""
echo "================================================================================"
echo "[3/4] Evaluating: checkpoint-224"
echo "================================================================================"
CHECKPOINT="${LORA_DIR}/checkpoint-224"
OUTPUT_DIR="${CHECKPOINT}/${DATE}_evaluation"
python inference.py --mode single \
    --model-path "$BASE_MODEL" \
    --lora-adapter "$CHECKPOINT" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --seed $SEED
echo "✓ checkpoint-224 evaluation complete: $OUTPUT_DIR"

# 4. Evaluate checkpoint-280
echo ""
echo "================================================================================"
echo "[4/4] Evaluating: checkpoint-280"
echo "================================================================================"
CHECKPOINT="${LORA_DIR}/checkpoint-280"
OUTPUT_DIR="${CHECKPOINT}/${DATE}_evaluation"
python inference.py --mode single \
    --model-path "$BASE_MODEL" \
    --lora-adapter "$CHECKPOINT" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --seed $SEED
echo "✓ checkpoint-280 evaluation complete: $OUTPUT_DIR"

echo ""
echo "================================================================================"
echo "Evaluation complete!"
echo "================================================================================"
echo "Results saved to:"
echo "  - Base: ${BASE_MODEL}/${DATE}_evaluation_32b_base/"
echo "  - CP-112: ${LORA_DIR}/checkpoint-112-backup/${DATE}_evaluation/"
echo "  - CP-224: ${LORA_DIR}/checkpoint-224/${DATE}_evaluation/"
echo "  - CP-280: ${LORA_DIR}/checkpoint-280/${DATE}_evaluation/"
