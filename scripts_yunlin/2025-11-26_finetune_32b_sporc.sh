#!/bin/bash

# =============================================================================
# Fine-tune Qwen3-VL-32B on SPoRC - 4,004 samples with dynamic prompts
#
# Changes from 2025-11-24 (894 samples):
#   - Dataset: 894 -> 4,004 samples (4.5x more data)
#   - LR: 4e-6 -> 1e-5 (more data allows faster training)
#   - NEFTune: 5.0 -> 2.0 (more data = less memorization risk)
#   - Warmup: 0.1 -> 0.03 (standard for ~250 steps)
#   - Save steps: 14 -> 25 (~10 checkpoints total)
#   - Prompt: dynamic word count range per sample
#
# Dataset: 4,004 excerpts with 1-6 images each (sporc_4004_samples_v2)
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
DATASET="sporc_4004_samples_v2%100"
NUM_SAMPLES=4004

# Output configuration with date prefix
DATE=$(date +%Y-%m-%d)
RUN_NAME="qwen3vl-32b-sporc-4004"
OUTPUT_DIR="/home/ubuntu/image-to-text/Qwen3-VL/finetuned_models/${DATE}_${RUN_NAME}"
LOG_FILE="${OUTPUT_DIR}/training.log"

# =============================================================================
# HYPERPARAMETERS (Gemini 3 Pro recommendations for larger dataset)
# =============================================================================
LR=1e-5                   # INCREASED from 4e-6 (more data allows faster training)
BATCH_SIZE=1              # Keep at 1 due to model size
GRAD_ACCUM_STEPS=4        # Effective batch = 32 (1 × 4 × 8 GPUs)
NUM_EPOCHS=2              # Keep at 2 (best result was epoch 1 previously)
WEIGHT_DECAY=0.1          # Keep high for stability
NEFTUNE_NOISE_ALPHA=2.0   # REDUCED from 5.0 (more data = less memorization risk)
WARMUP_RATIO=0.03         # REDUCED from 0.1 (standard for ~250 steps)

# LoRA configuration - keep rank 16, proven to work
USE_LORA=True
LORA_R=16
LORA_ALPHA=32             # 2x rank
LORA_DROPOUT=0.05
# Attention + MLP
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Training entry point
ENTRY_FILE=qwen-vl-finetune/qwenvl/train/train_qwen.py

# DeepSpeed config
DEEPSPEED_CONFIG=qwen-vl-finetune/scripts/zero3.json

# Calculate steps
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE))
STEPS_PER_EPOCH=$((NUM_SAMPLES / EFFECTIVE_BATCH))
TOTAL_STEPS=$((STEPS_PER_EPOCH * NUM_EPOCHS))
# Save at: mid-epoch1 (~62), end-epoch1 (~125), mid-epoch2 (~187), end-epoch2 (~250)
SAVE_STEPS=62

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
echo "Qwen3-VL-32B Fine-tuning (4,004 samples with dynamic prompts)"
echo "================================================================================"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET} (${NUM_SAMPLES} samples)"
echo ""
echo "Changes from previous (894 samples):"
echo "  - Dataset: 894 -> ${NUM_SAMPLES} samples (4.5x more)"
echo "  - LR: 4e-6 -> ${LR} (increased)"
echo "  - NEFTune: 5.0 -> ${NEFTUNE_NOISE_ALPHA} (reduced)"
echo "  - Warmup: 0.1 -> ${WARMUP_RATIO} (reduced)"
echo "  - Save steps: 14 -> ${SAVE_STEPS}"
echo "  - Prompt: dynamic word count range per sample"
echo ""
echo "Hyperparameters:"
echo "  Learning rate: ${LR}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "  Effective batch size: ${EFFECTIVE_BATCH}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Steps per epoch: ~${STEPS_PER_EPOCH}"
echo "  Total steps: ~${TOTAL_STEPS}"
echo "  Weight decay: ${WEIGHT_DECAY}"
echo "  NEFTune alpha: ${NEFTUNE_NOISE_ALPHA}"
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
