#!/bin/bash

# =============================================================================
# Fine-tune Qwen3-VL-8B on SPoRC - CONSERVATIVE VERSION v2
#
# Dataset: 894 complete excerpts (mean: 824 words, median: 808 words)
# Prompt: FULL DETAILED prompt from Nov 9 Test 1 (with examples)
# Method: Conservative LoRA - attention ONLY, frozen MLP to prevent mode collapse
# Hardware: 4 GPUs (effective batch size: 16)
# Changes from v1: 1 epoch, lower LR, smaller rank, no MLP, more checkpoints
# =============================================================================

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}  # Number of GPUs
NNODES=${WORLD_SIZE:-1}

# NOTE: Set CUDA_VISIBLE_DEVICES before running this script
# For GPUs 4-7: export CUDA_VISIBLE_DEVICES=4,5,6,7 && bash finetune_8b_sporc.sh

# Model configuration
MODEL_PATH="/home/ubuntu/LLM/qwen3-vl-8b"

# Dataset configuration
DATASET="sporc_excerpt_detailed_prompt%100"  # Use 100% of 894 excerpts with FULL detailed prompt

# Output configuration with date prefix
DATE=$(date +%Y-%m-%d)
RUN_NAME="qwen3vl-8b-sporc-v2-conservative"
OUTPUT_DIR="./finetuned_models/${DATE}_${RUN_NAME}"
LOG_FILE="${OUTPUT_DIR}/training.log"

# Training hyperparameters - CONSERVATIVE to prevent mode collapse
# Effective batch size: 1 × 4 × 4 GPUs = 16 (reasonable for fine-tuning)
LR=1e-5                   # REDUCED from 5e-5 to prevent overfitting
BATCH_SIZE=1              # Safe with 5 images per sample
GRAD_ACCUM_STEPS=4        # Effective batch = 16
NUM_EPOCHS=1              # REDUCED from 3 to 1 epoch only

# LoRA configuration - CONSERVATIVE (attention only, no MLP)
USE_LORA=True
LORA_R=32                 # REDUCED from 64 to 32
LORA_ALPHA=32             # Match rank (was 128)
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"  # ATTENTION ONLY (removed gate/up/down MLP)

# Image resolution
MAX_PIXELS=50176   # Standard Qwen resolution
MIN_PIXELS=784

# Training entry point
ENTRY_FILE=qwen-vl-finetune/qwenvl/train/train_qwen.py

# DeepSpeed config (ZeRO-3 for memory efficiency)
DEEPSPEED_CONFIG=qwen-vl-finetune/scripts/zero3.json

# Training arguments - CONSERVATIVE configuration
args="
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
    --per_device_eval_batch_size $((BATCH_SIZE*2)) \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --max_pixels ${MAX_PIXELS} \
    --min_pixels ${MIN_PIXELS} \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 25 \
    --save_total_limit 5 \
    --learning_rate ${LR} \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 6144 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${RUN_NAME} \
    --report_to none"

# Print configuration
echo "================================================================================"
echo "Qwen3-VL-8B CONSERVATIVE Fine-tuning (v2) - Preventing Mode Collapse"
echo "================================================================================"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET} (894 samples with FULL detailed prompt)"
echo ""
echo "Conservative Changes from v1:"
echo "  ✓ Epochs: 1 (was 3)"
echo "  ✓ Learning rate: ${LR} (was 5e-5)"
echo "  ✓ LoRA rank: ${LORA_R} (was 64)"
echo "  ✓ LoRA alpha: ${LORA_ALPHA} (was 128)"
echo "  ✓ LoRA targets: attention only (was attention + MLP)"
echo "  ✓ MLP training: FROZEN (was trained)"
echo "  ✓ data_flatten: False (was True)"
echo "  ✓ Warmup: 10% (was 3%)"
echo "  ✓ Weight decay: 0.01 (was 0)"
echo "  ✓ Save every: 25 steps (was 100)"
echo "  ✓ Keep checkpoints: 5 (was 2)"
echo "  ✓ Max seq length: 6144 (was 8192)"
echo ""
echo "Training Configuration:"
echo "  GPUs: ${NPROC_PER_NODE}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "  Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE))"
echo "  Total steps: ~56 (894 samples / 16 batch)"
echo "  Warmup steps: ~6 (10% of 56)"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "================================================================================"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Save a copy of this script and training code for reproducibility
echo "Saving training configuration..."
cp "$0" "${OUTPUT_DIR}/$(basename $0)"
cp "${ENTRY_FILE}" "${OUTPUT_DIR}/train_qwen.py"
echo "✓ Saved: $(basename $0) -> ${OUTPUT_DIR}/"
echo "✓ Saved: train_qwen.py -> ${OUTPUT_DIR}/"
echo ""

# Launch training with logging
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${ENTRY_FILE} ${args} 2>&1 | tee ${LOG_FILE}
