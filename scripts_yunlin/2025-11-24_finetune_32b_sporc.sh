#!/bin/bash

# =============================================================================
# Fine-tune Qwen3-VL-32B on SPoRC - v2 with Gemini recommendations
#
# Changes from v1 (11/22):
#   - LR: 2e-5 -> 4e-6 (5x lower to prevent overfitting)
#   - Weight decay: 0.01 -> 0.1 (10x higher for regularization)
#   - Epochs: 5 -> 2 (stop early)
#   - NEFTune: enabled (noise_alpha=5) to prevent memorization
#   - Effective batch: 16 -> 32 (more stable gradients)
#
# Dataset: 894 complete excerpts (sporc_excerpt%100)
# Hardware: 8 GPUs
# =============================================================================

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}

# Model configuration
MODEL_PATH="/home/ubuntu/LLM/qwen3-vl-32b"

# Dataset configuration
DATASET="sporc_excerpt%100"

# Output configuration with date prefix
DATE=$(date +%Y-%m-%d)
RUN_NAME="qwen3vl-32b-sporc-v2"
OUTPUT_DIR="/home/ubuntu/image-to-text/Qwen3-VL/finetuned_models/${DATE}_${RUN_NAME}"
LOG_FILE="${OUTPUT_DIR}/training.log"

# =============================================================================
# UPDATED HYPERPARAMETERS (Gemini recommendations + user preferences)
# =============================================================================
LR=4e-6                   # REDUCED from 2e-5 (5x lower)
BATCH_SIZE=1              # Keep at 1 due to model size
GRAD_ACCUM_STEPS=4        # Increased: effective batch = 32 (1 × 4 × 8 GPUs)
NUM_EPOCHS=2              # REDUCED from 5 to 2 (stop early)
WEIGHT_DECAY=0.1          # INCREASED from 0.01 (10x for regularization)
NEFTUNE_NOISE_ALPHA=5.0   # NEW: prevents memorization on small datasets

# LoRA configuration - keep rank 16, include MLP
USE_LORA=True
LORA_R=16                 # Keep at 16
LORA_ALPHA=32             # 2x rank
LORA_DROPOUT=0.05
# Attention + MLP (user requested MLP to be included)
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Training entry point
ENTRY_FILE=qwen-vl-finetune/qwenvl/train/train_qwen.py

# DeepSpeed config
DEEPSPEED_CONFIG=qwen-vl-finetune/scripts/zero3.json

# Training arguments
ARGS="
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_use ${DATASET} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --use_lora ${USE_LORA} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target_modules ${LORA_TARGET_MODULES} \
    --bf16 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 14 \
    --save_total_limit 10 \
    --learning_rate ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio 0.1 \
    --max_grad_norm 0.3 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --neftune_noise_alpha ${NEFTUNE_NOISE_ALPHA} \
    --run_name ${RUN_NAME} \
    --report_to none"

# Print configuration
echo "================================================================================"
echo "Qwen3-VL-32B Fine-tuning v2 (Gemini recommendations)"
echo "================================================================================"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET} (894 samples)"
echo ""
echo "v2 Changes from v1:"
echo "  - LR: 2e-5 -> ${LR} (5x lower)"
echo "  - Weight decay: 0.01 -> ${WEIGHT_DECAY} (10x higher)"
echo "  - Epochs: 5 -> ${NUM_EPOCHS} (stop early)"
echo "  - NEFTune: ${NEFTUNE_NOISE_ALPHA} (prevents memorization)"
echo "  - Effective batch: 16 -> $((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE)) (more stable)"
echo "  - max_grad_norm: 1.0 -> 0.3 (tighter clipping)"
echo "  - Save every 14 steps (~4 checkpoints)"
echo ""
echo "Hyperparameters:"
echo "  Learning rate: ${LR}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "  Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE))"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Total steps: ~$((894 * NUM_EPOCHS / (BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE)))"
echo ""
echo "LoRA Configuration:"
echo "  LoRA rank (r): ${LORA_R}"
echo "  LoRA alpha: ${LORA_ALPHA}"
echo "  Target modules: ${LORA_TARGET_MODULES}"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "================================================================================"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Save a copy of this script for reproducibility
echo "Saving training configuration..."
cp "$0" "${OUTPUT_DIR}/$(basename $0)"
echo "Saved: $(basename $0) -> ${OUTPUT_DIR}/"
echo ""

# Launch training with logging
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${ENTRY_FILE} ${ARGS} 2>&1 | tee ${LOG_FILE}

echo ""
echo "================================================================================"
echo "Training completed!"
echo "LoRA adapter saved to: ${OUTPUT_DIR}"
echo ""
echo "To run inference with LoRA adapter:"
echo "  cd /home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin"
echo "  python inference.py --mode interactive --lora-adapter ${OUTPUT_DIR}"
echo "================================================================================"
