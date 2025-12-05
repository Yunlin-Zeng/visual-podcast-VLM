#!/bin/bash

# =============================================================================
# Fine-tune Qwen3-VL-235B with PRECOMPUTED vision embeddings
#
# Goal: Train LLM-only WITHOUT loading vision encoder and MLP
# Expected memory savings: 30-50GB per GPU (parameters + activations + optimizer)
# Hardware: 8× A100 80GB
#
# This is attempt #17 - Phase 2 of precomputed embeddings approach
# Previous approach (attempt #16): Frozen components still loaded → OOM
# New approach: Don't load vision components at all → use precomputed embeddings
# =============================================================================

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}  # Number of GPUs

# Model configuration - USE LLM-ONLY CHECKPOINT (vision weights excluded)
MODEL_PATH="/home/ubuntu/LLM/qwen3-vl-235b-llm-only"

# Dataset configuration - USE PRECOMPUTED EMBEDDINGS
DATASET="sporc_excerpt%100"  # Use 100% of our 894 complete excerpts
PRECOMPUTED_DIR="./precomputed_embeddings_235b_sporc"

# Output configuration
RUN_NAME="qwen3vl-235b-lora-precomputed-attempt17"
OUTPUT_DIR="./finetuned_models/${RUN_NAME}"

# Training hyperparameters - VERY CONSERVATIVE
LR=5e-6
BATCH_SIZE=1  # Minimum batch size per device
GRAD_ACCUM_STEPS=4  # Effective batch size = 1 × 4 × 8 GPUs = 32

# Image resolution (not used for precomputed embeddings, but keep for compatibility)
MAX_PIXELS=50176   # 224×224 pixels
MIN_PIXELS=784     # 28×28 pixels

# Training entry point - CUSTOM SCRIPT FOR PRECOMPUTED EMBEDDINGS
ENTRY_FILE=qwen-vl-finetune/qwenvl/train/train_qwen_precomputed.py

# DeepSpeed config - use ZeRO-2 (MoE models don't support ZeRO-3)
DEEPSPEED_CONFIG=qwen-vl-finetune/scripts/zero2.json

# LoRA configuration - CONSERVATIVE settings
USE_LORA=True
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"  # Attention only

# Training arguments
ARGS="
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --precomputed_embeddings_dir ${PRECOMPUTED_DIR} \
    --use_precomputed_embeddings True \
    --dataset_use ${DATASET} \
    --data_flatten False \
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
echo "Qwen3-VL-235B LoRA Fine-tuning - Attempt #17 (Precomputed Embeddings - Phase 2)"
echo "=============================================================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET} (894 samples)"
echo "Precomputed embeddings: ${PRECOMPUTED_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Hardware:"
echo "  GPUs: ${NPROC_PER_NODE} × A100 80GB"
echo "  DeepSpeed: ZeRO-2 (MoE compatible)"
echo ""
echo "Memory Optimization (Phase 2):"
echo "  Vision encoder: ❌ NOT LOADED (saves ~10-15GB parameters)"
echo "  MLP projector: ❌ NOT LOADED (saves ~5-10GB parameters)"
echo "  LLM: ✅ Trainable (LoRA adapters only)"
echo "  Vision embeddings: ✅ Loaded from precomputed files"
echo "  Expected total memory savings: ~30-50GB per GPU vs. full loading"
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
