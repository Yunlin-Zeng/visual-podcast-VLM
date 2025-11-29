#!/bin/bash

# =============================================================================
# Fine-tune Qwen3-VL-32B on SPoRC - SUPER CONSERVATIVE
#
# Exactly matches 894-sample experiment setup:
#   - Exactly 5 images per sample (same as 894-sample)
#   - Fixed prompt: "around 800 words" (no dynamic word count)
#   - Same prompt template as 894-sample experiment
#   - Same hyperparameters as 894-sample experiment
#
# Dataset: 3737 samples (sporc_super_conservative)
# Hardware: 8 GPUs
# =============================================================================

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}

# Model configuration
MODEL_PATH="/home/ubuntu/LLM/qwen3-vl-32b"

# Dataset configuration - super conservative (exactly 5 images, 800 words)
DATASET="sporc_super_conservative%100"

# Output configuration with date prefix
DATE=$(date +%Y-%m-%d)
RUN_NAME="qwen3vl-32b-sporc-super-conservative"
OUTPUT_DIR="/home/ubuntu/image-to-text/Qwen3-VL/finetuned_models/${DATE}_${RUN_NAME}"
LOG_FILE="${OUTPUT_DIR}/training.log"

# =============================================================================
# HYPERPARAMETERS - EXACTLY SAME AS 894-SAMPLE RUN
# =============================================================================
LR=4e-6                   # SAME as 894-sample
BATCH_SIZE=1              # Keep at 1 due to model size
GRAD_ACCUM_STEPS=4        # Effective batch = 32 (1 × 4 × 8 GPUs)
NUM_EPOCHS=1              # 1 epoch first, can continue later
WEIGHT_DECAY=0.1          # SAME as 894-sample
WARMUP_RATIO=0.1          # SAME as 894-sample
NEFTUNE_NOISE_ALPHA=5.0   # SAME as 894-sample (prevents memorization)

# Save steps calculation:
# 3737 samples / 32 effective batch = 116.78 steps per epoch
# Save at ~half epoch (58) and end (~117)
SAVE_STEPS=58

# LoRA configuration - SAME as 894-sample
USE_LORA=True
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
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
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 10 \
    --learning_rate ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio ${WARMUP_RATIO} \
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
echo "Qwen3-VL-32B Fine-tuning - SUPER CONSERVATIVE"
echo "================================================================================"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET} (3737 samples)"
echo ""
echo "Setup (EXACTLY matches 894-sample experiment):"
echo "  - Exactly 5 images per sample"
echo "  - Fixed prompt: 'around 800 words'"
echo "  - 'Speaker 1:, Speaker 2:' format"
echo "  - 'without introductions or sign-offs'"
echo ""
echo "Hyperparameters (SAME as 894-sample):"
echo "  Learning rate: ${LR}"
echo "  NEFTune alpha: ${NEFTUNE_NOISE_ALPHA}"
echo "  Weight decay: ${WEIGHT_DECAY}"
echo "  Warmup ratio: ${WARMUP_RATIO}"
echo "  max_grad_norm: 0.3"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "  Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE))"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Steps per epoch: ~$((3737 / (BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE)))"
echo "  Save steps: ${SAVE_STEPS}"
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
