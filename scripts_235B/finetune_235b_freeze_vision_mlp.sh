#!/bin/bash

# =============================================================================
# Fine-tune Qwen3-VL-235B with FROZEN vision encoder and MLP
#
# Goal: Train LLM-only by freezing vision components to reduce memory usage
# Expected memory savings: ~25-40GB per GPU (activations + optimizer states)
# Hardware: 8× A100 80GB
#
# This is attempt #16 - previous attempts 1-15 all failed with OOM
# Key difference: Freeze vision encoder + MLP to save memory
# =============================================================================

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}  # Number of GPUs

# Model configuration
MODEL_PATH="/home/ubuntu/LLM/qwen3-vl-235b"

# Dataset configuration
DATASET="sporc_excerpt%100"  # Use 100% of our 894 complete excerpts

# Output configuration
RUN_NAME="qwen3vl-235b-lora-freeze-vision-attempt16"
OUTPUT_DIR="./finetuned_models/${RUN_NAME}"

# Training hyperparameters - VERY CONSERVATIVE to avoid OOM
# - Smallest possible batch size
# - Gradient accumulation for effective batch size
# - Lower learning rate since we're only training LLM
LR=5e-6
BATCH_SIZE=1  # Minimum batch size per device
GRAD_ACCUM_STEPS=4  # Effective batch size = 1 × 4 × 8 GPUs = 32

# Image resolution (same as successful 32B training)
MAX_PIXELS=50176   # 224×224 pixels
MIN_PIXELS=784     # 28×28 pixels

# Training entry point
ENTRY_FILE=qwen-vl-finetune/qwenvl/train/train_qwen.py

# DeepSpeed config - use ZeRO-2 (MoE models don't support ZeRO-3)
DEEPSPEED_CONFIG=qwen-vl-finetune/scripts/zero2.json

# LoRA configuration - CONSERVATIVE settings
# - Smaller rank to reduce memory
# - Only apply to attention layers (not FFN)
USE_LORA=True
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"  # Attention only

# Training arguments
ARGS="
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_use ${DATASET} \
    --data_flatten True \
    --data_packing False \
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
    --run_name ${RUN_NAME} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --max_pixels ${MAX_PIXELS} \
    --min_pixels ${MIN_PIXELS} \
    --learning_rate ${LR} \
    --optim adamw_torch \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --model_max_length 8192 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 1 \
    --report_to none \
    --eval_strategy no \
    --gradient_checkpointing True \
    --dataloader_num_workers 4
"

# Print configuration
echo "=============================================================================="
echo "Qwen3-VL-235B LoRA Fine-tuning - Attempt #16 (Freeze Vision Components)"
echo "=============================================================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET} (894 samples)"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Hardware:"
echo "  GPUs: ${NPROC_PER_NODE} × A100 80GB"
echo "  DeepSpeed: ZeRO-2 (MoE compatible)"
echo ""
echo "Memory Optimization:"
echo "  Vision encoder: ❌ FROZEN (saves ~15-25GB activations)"
echo "  MLP projector: ❌ FROZEN (saves ~10-15GB optimizer states)"
echo "  LLM: ✅ Trainable (LoRA adapters only)"
echo "  Expected memory savings: ~25-40GB per GPU"
echo ""
echo "Hyperparameters:"
echo "  Learning rate: ${LR}"
echo "  Batch size per device: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "  Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE))"
echo "  Epochs: 3"
echo "  Steps per epoch: ~$((894 / (BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE)))"
echo "  Total steps: ~$((894 * 3 / (BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE)))"
echo ""
echo "LoRA Configuration:"
echo "  LoRA enabled: ${USE_LORA}"
echo "  LoRA rank (r): ${LORA_R}"
echo "  LoRA alpha: ${LORA_ALPHA}"
echo "  LoRA dropout: ${LORA_DROPOUT}"
echo "  Target modules: ${LORA_TARGET_MODULES} (attention only)"
echo ""
echo "Image resolution:"
echo "  Max: ${MAX_PIXELS} pixels (224×224)"
echo "  Min: ${MIN_PIXELS} pixels (28×28)"
echo "=============================================================================="
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${ENTRY_FILE} ${ARGS}

echo ""
echo "=============================================================================="
echo "Training completed!"
echo "LoRA adapter saved to: ${OUTPUT_DIR}"
echo "=============================================================================="
