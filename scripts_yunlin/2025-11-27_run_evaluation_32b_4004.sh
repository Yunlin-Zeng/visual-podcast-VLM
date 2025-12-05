#!/bin/bash
# Run evaluation on 32B model trained with 4004 samples
# Order: 124, 248, 62, 186

set -e

# Configuration
BASE_MODEL="/home/ubuntu/LLM/qwen3-vl-32b"
LORA_DIR="/home/ubuntu/image-to-text/Qwen3-VL/finetuned_models/2025-11-26_qwen3vl-32b-sporc-4004"
SAMPLES_DIR="/home/ubuntu/image-to-text/Qwen3-VL/inference_samples"
PROMPT_FILE="/home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin/evaluation_prompt.txt"
SEED=0
DATE=$(date +%Y-%m-%d)

cd /home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin

echo "================================================================================"
echo "32B Model Evaluation (4004 samples, 4 checkpoints, seed=$SEED)"
echo "================================================================================"
echo "Base model: $BASE_MODEL"
echo "LoRA dir: $LORA_DIR"
echo "Samples: $SAMPLES_DIR"
echo "Seed: $SEED"
echo "================================================================================"

# 1. Evaluate checkpoint-124 (end epoch 1)
echo ""
echo "================================================================================"
echo "[1/4] Evaluating: checkpoint-124 (end epoch 1)"
echo "================================================================================"
CHECKPOINT="${LORA_DIR}/checkpoint-124"
OUTPUT_DIR="${CHECKPOINT}/${DATE}_evaluation"
python inference.py --mode single \
    --model-path "$BASE_MODEL" \
    --lora-adapter "$CHECKPOINT" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --seed $SEED
echo "checkpoint-124 evaluation complete: $OUTPUT_DIR"

# 2. Evaluate checkpoint-248 (end epoch 2)
echo ""
echo "================================================================================"
echo "[2/4] Evaluating: checkpoint-248 (end epoch 2)"
echo "================================================================================"
CHECKPOINT="${LORA_DIR}/checkpoint-248"
OUTPUT_DIR="${CHECKPOINT}/${DATE}_evaluation"
python inference.py --mode single \
    --model-path "$BASE_MODEL" \
    --lora-adapter "$CHECKPOINT" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --seed $SEED
echo "checkpoint-248 evaluation complete: $OUTPUT_DIR"

# 3. Evaluate checkpoint-62 (mid epoch 1)
echo ""
echo "================================================================================"
echo "[3/4] Evaluating: checkpoint-62 (mid epoch 1)"
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

# 4. Evaluate checkpoint-186 (mid epoch 2)
echo ""
echo "================================================================================"
echo "[4/4] Evaluating: checkpoint-186 (mid epoch 2)"
echo "================================================================================"
CHECKPOINT="${LORA_DIR}/checkpoint-186"
OUTPUT_DIR="${CHECKPOINT}/${DATE}_evaluation"
python inference.py --mode single \
    --model-path "$BASE_MODEL" \
    --lora-adapter "$CHECKPOINT" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --seed $SEED
echo "checkpoint-186 evaluation complete: $OUTPUT_DIR"

echo ""
echo "================================================================================"
echo "Evaluation complete!"
echo "================================================================================"
echo "Results saved to:"
echo "  - CP-124: ${LORA_DIR}/checkpoint-124/${DATE}_evaluation/"
echo "  - CP-248: ${LORA_DIR}/checkpoint-248/${DATE}_evaluation/"
echo "  - CP-62:  ${LORA_DIR}/checkpoint-62/${DATE}_evaluation/"
echo "  - CP-186: ${LORA_DIR}/checkpoint-186/${DATE}_evaluation/"
