#!/bin/bash

# Set PyTorch memory allocator to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =============================================================================
# Fine-tune Qwen3-VL-235B with On-the-fly 4-bit Quantization
#
# Dataset: 894 complete excerpts with 235B-aligned prompt (topics/themes focused)
# Method: Conservative LoRA - attention ONLY, frozen MLP to prevent mode collapse
# Hardware: 8 GPUs (effective batch size: 16)
# Model: Original BF16, quantized to 4-bit NF4 during loading
#
# CRITICAL: Qwen3-VL-235B is MoE model - MUST use ZeRO-2 (NOT ZeRO-3)
# =============================================================================

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}  # Number of GPUs
NNODES=${WORLD_SIZE:-1}

# Model configuration - ORIGINAL BF16 MODEL (will quantize on load)
MODEL_PATH="/home/ubuntu/LLM/qwen3-vl-235b-a22b"  # Original BF16 model

# Dataset configuration
DATASET="sporc_235b_aligned%100"  # Use 100% of 894 excerpts with 235B-aligned prompt

# Output configuration with date prefix
DATE=$(date +%Y-%m-%d)
RUN_NAME="qwen3vl-235b-load4bit-sporc"
OUTPUT_DIR="./finetuned_models/${DATE}_${RUN_NAME}"
LOG_FILE="${OUTPUT_DIR}/training.log"

# Training hyperparameters - CONSERVATIVE to prevent mode collapse
# Effective batch size: 1 × 2 × 8 GPUs = 16
LR=1e-6                   # Scaled from 8B's 1e-5 for larger model
BATCH_SIZE=1              # Safe with 5 images per sample
GRAD_ACCUM_STEPS=2        # Effective batch = 16
NUM_EPOCHS=1              # Conservative: 1 epoch only

# LoRA configuration - ULTRA CONSERVATIVE (attention only, no MLP, reduced rank)
USE_LORA=True
LORA_R=8                  # Ultra conservative for memory (quarter of 8B's 32)
LORA_ALPHA=8              # Match rank for 1:1 ratio
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"  # ATTENTION ONLY (no gate/up/down MLP)

# 4-bit quantization configuration
LOAD_IN_4BIT=True
BNB_4BIT_COMPUTE_DTYPE="bfloat16"
BNB_4BIT_QUANT_TYPE="nf4"
BNB_4BIT_USE_DOUBLE_QUANT=True

# Image resolution - standard resolution
MAX_PIXELS=50176   # Standard Qwen resolution (512×28×28)
MIN_PIXELS=784

# Training entry point
ENTRY_FILE=qwen-vl-finetune/qwenvl/train/train_qwen.py

# DeepSpeed config - MUST use ZeRO-2 for MoE models (NOT ZeRO-3)
# Using CPU offload version to reduce GPU memory pressure
DEEPSPEED_CONFIG=qwen-vl-finetune/scripts/zero2_offload.json

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
    --load_in_4bit ${LOAD_IN_4BIT} \
    --bnb_4bit_compute_dtype ${BNB_4BIT_COMPUTE_DTYPE} \
    --bnb_4bit_quant_type ${BNB_4BIT_QUANT_TYPE} \
    --bnb_4bit_use_double_quant ${BNB_4BIT_USE_DOUBLE_QUANT} \
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
    --save_steps 16 \
    --save_total_limit 5 \
    --learning_rate ${LR} \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 6144 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${RUN_NAME} \
    --report_to none"

# Print configuration
echo "================================================================================"
echo "Qwen3-VL-235B (On-the-fly 4-bit Quantization) Fine-tuning"
echo "================================================================================"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET} (894 samples with 235B-aligned prompt)"
echo ""
echo "On-the-fly 4-bit Quantization:"
echo "  ✓ Approach: Load BF16 model → quantize to NF4 during loading"
echo "  ✓ Quantization type: NF4 (4-bit)"
echo "  ✓ Compute dtype: bfloat16"
echo "  ✓ Double quantization: enabled"
echo "  ✓ Expected GPU memory: ~15GB per GPU (vs 59GB BF16)"
echo ""
echo "Ultra-Conservative Configuration:"
echo "  ✓ Epochs: 1"
echo "  ✓ Learning rate: ${LR} (scaled from 8B's 1e-5)"
echo "  ✓ LoRA rank: ${LORA_R} (quarter of 8B's 32 - ultra conservative)"
echo "  ✓ LoRA alpha: ${LORA_ALPHA} (1:1 ratio)"
echo "  ✓ LoRA targets: attention only (q/k/v/o)"
echo "  ✓ MLP training: FROZEN"
echo "  ✓ Vision-language projector: FROZEN"
echo "  ✓ Image resolution: ${MAX_PIXELS} pixels (standard)"
echo "  ✓ Memory fragmentation: expandable_segments enabled"
echo "  ✓ data_flatten: False"
echo "  ✓ Warmup: 10%"
echo "  ✓ Weight decay: 0.01"
echo "  ✓ Gradient clipping: 1.0"
echo "  ✓ Save every: 16 steps (~4 times per epoch)"
echo "  ✓ Keep checkpoints: 5"
echo "  ✓ Max seq length: 6144"
echo ""
echo "Training Configuration:"
echo "  GPUs: ${NPROC_PER_NODE}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "  Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE))"
echo "  Total steps: ~56 (894 samples / 16 batch)"
echo "  Warmup steps: ~6 (10% of 56)"
echo "  Expected checkpoints: checkpoint-16, checkpoint-32, checkpoint-48, checkpoint-56"
echo ""
echo "CRITICAL: Using ZeRO-2 with CPU offload (NOT ZeRO-3) - MoE models incompatible with ZeRO-3"
echo "  ✓ Optimizer states offloaded to CPU to reduce GPU memory pressure"
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
cp "scripts_yunlin/2025-11-11_sporc_prompt_235b.py" "${OUTPUT_DIR}/2025-11-11_sporc_prompt_235b.py" 2>/dev/null || echo "Note: 2025-11-11_sporc_prompt_235b.py not found"
echo "✓ Saved: $(basename $0) -> ${OUTPUT_DIR}/"
echo "✓ Saved: train_qwen.py -> ${OUTPUT_DIR}/"
echo ""

# Launch training with logging
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${ENTRY_FILE} ${args} 2>&1 | tee ${LOG_FILE}
