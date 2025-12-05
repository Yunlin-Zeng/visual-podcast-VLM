#!/bin/bash

# =============================================================================
# Fine-tune Qwen3-VL-235B on SPoRC - CONSERVATIVE V2 (based on working script)
#
# Dataset: 894 complete excerpts with 235B-aligned prompt (topics/themes focused)
# Method: Conservative LoRA - attention ONLY, frozen MLP AND vision encoder
# Hardware: 8× A100 80GB
# Key: Using data_flatten=True from previous successful 235B training
#
# IMPORTANT: Qwen3-VL-235B is MoE and does NOT support DeepSpeed ZeRO-3
#            Must use ZeRO-2 or no DeepSpeed
# =============================================================================

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}  # Number of GPUs

# Model configuration
MODEL_PATH="/home/ubuntu/LLM/qwen3-vl-235b-a22b"  # Symlink with "a" to trigger MoE detection

# Dataset configuration
DATASET="sporc_235b_aligned%100"  # Use 100% of 894 excerpts with 235B-aligned prompt

# Output configuration with date prefix
DATE=$(date +%Y-%m-%d)
RUN_NAME="qwen3vl-235b-sporc-conservative-v2"
OUTPUT_DIR="./finetuned_models/${DATE}_${RUN_NAME}"

# Training hyperparameters - CONSERVATIVE
LR=1e-6                   # More conservative than previous 2e-6
BATCH_SIZE=1              # Safe with 5 images per sample
GRAD_ACCUM_STEPS=2        # Effective batch = 16

# Image resolution (same as inference: 512×32×32)
# max_pixels = 512 * 32 * 32 = 524,288 pixels
MAX_PIXELS=50176   # Qwen format: 50176 = 1792 × 28 = 512×32×32 converted
MIN_PIXELS=784     # Minimum resolution

# Training entry point
ENTRY_FILE=qwen-vl-finetune/qwenvl/train/train_qwen.py

# DeepSpeed config (use ZeRO-2, NOT ZeRO-3 for MoE models)
DEEPSPEED_CONFIG=qwen-vl-finetune/scripts/zero2.json

# LoRA configuration - CONSERVATIVE (attention only, no MLP)
USE_LORA=True
LORA_R=16                 # Conservative for 235B
LORA_ALPHA=16             # Match rank for 1:1 ratio
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"  # ATTENTION ONLY (no gate/up/down MLP)

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
    --num_train_epochs 1 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --max_pixels ${MAX_PIXELS} \
    --min_pixels ${MIN_PIXELS} \
    --learning_rate ${LR} \
    --optim adamw_torch \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --model_max_length 6144 \
    --save_strategy steps \
    --save_steps 16 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --report_to none \
    --eval_strategy no \
    --gradient_checkpointing True \
    --dataloader_num_workers 4
"

# Print configuration
echo "================================================================================"
echo "Qwen3-VL-235B CONSERVATIVE Fine-tuning V2 (data_flatten=True)"
echo "================================================================================"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET} (894 samples with 235B-aligned prompt)"
echo ""
echo "New 235B-Aligned Prompt:"
echo "  ✓ Focus: Topics/themes related to images (not pure visual description)"
echo "  ✓ Style: Casual, natural conversation"
echo "  ✓ Target: 800 words (matches dataset average)"
echo "  ✓ Goal: Reduce visual cataloging, increase topic discussion"
echo ""
echo "Conservative Configuration (scaled from 8B v2):"
echo "  ✓ Epochs: 1"
echo "  ✓ Learning rate: ${LR} (scaled from 8B's 1e-5)"
echo "  ✓ LoRA rank: ${LORA_R} (half of 8B's 32)"
echo "  ✓ LoRA alpha: ${LORA_ALPHA} (1:1 ratio)"
echo "  ✓ LoRA targets: attention only (q/k/v/o)"
echo "  ✓ MLP training: FROZEN"
echo "  ✓ Vision encoder: FROZEN"
echo "  ✓ data_flatten: True (KEY: from previous successful 235B training)"
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
echo "CRITICAL: Using ZeRO-2 (NOT ZeRO-3) - MoE models incompatible with ZeRO-3"
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
cp "scripts_yunlin/2025-11-11_sporc_prompt_235b.py" "${OUTPUT_DIR}/2025-11-11_sporc_prompt_235b.py"
echo "✓ Saved: $(basename $0) -> ${OUTPUT_DIR}/"
echo "✓ Saved: train_qwen.py -> ${OUTPUT_DIR}/"
echo "✓ Saved: 2025-11-11_sporc_prompt_235b.py -> ${OUTPUT_DIR}/"
echo ""

# Launch training with logging
LOG_FILE="${OUTPUT_DIR}/training.log"
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
echo "  python inference.py --mode interactive --lora-adapter ../${OUTPUT_DIR} --seed 0"
echo ""
echo "Note: We load base model + LoRA adapter on-the-fly (no merging needed)"
echo "================================================================================"
