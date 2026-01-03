#!/bin/bash

# =============================================================================
# Fine-tune Qwen3-VL-8B on SPoRC podcast excerpts with WORKING prompt format
#
# Dataset: 894 complete excerpts (mean: 824 words, median: 808 words)
# Prompt: Working Q&A format with 800-word target (tested and validated)
# Method: LoRA fine-tuning (tune LLM + MLP with LoRA, freeze vision encoder)
# Hardware: 4 GPUs (effective batch size: 16)
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
DATASET="sporc_excerpt_working_prompt%100"  # Use 100% of 894 excerpts with working prompt

# Output configuration
RUN_NAME="qwen3vl-8b-sporc-working-prompt"
OUTPUT_DIR="./finetuned_models/${RUN_NAME}"
LOG_FILE="${OUTPUT_DIR}/training.log"

# Training hyperparameters (LoRA fine-tuning for 8B with 5 images/sample)
# LoRA allows higher learning rate and is more memory efficient
# Effective batch size: 1 × 4 × 4 GPUs = 16 (reasonable for fine-tuning)
LR=5e-5                   # Higher LR for LoRA (vs 1e-6 for full FT)
BATCH_SIZE=1              # Safe with 5 images per sample
GRAD_ACCUM_STEPS=4        # Effective batch = 32
NUM_EPOCHS=3

# LoRA configuration (recommended by Gemini/GPT5)
USE_LORA=True
LORA_R=64                 # LoRA rank
LORA_ALPHA=128            # LoRA alpha (2x rank)
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Image resolution
MAX_PIXELS=50176   # Standard Qwen resolution
MIN_PIXELS=784

# Training entry point
ENTRY_FILE=qwen-vl-finetune/qwenvl/train/train_qwen.py

# DeepSpeed config (ZeRO-3 for memory efficiency)
DEEPSPEED_CONFIG=qwen-vl-finetune/scripts/zero3.json

# Training arguments
args="
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_use ${DATASET} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
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
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate ${LR} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${RUN_NAME} \
    --report_to none"

# Print configuration
echo "=========================================="
echo "Qwen3-VL-8B LoRA Fine-tuning Configuration"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET} (894 samples)"
echo "Method: LoRA (r=${LORA_R}, alpha=${LORA_ALPHA})"
echo "GPUs: ${NPROC_PER_NODE} (using GPUs 4-7)"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo "Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE))"
echo "Learning rate: ${LR}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Launch training with logging
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${ENTRY_FILE} ${args} 2>&1 | tee ${LOG_FILE}
