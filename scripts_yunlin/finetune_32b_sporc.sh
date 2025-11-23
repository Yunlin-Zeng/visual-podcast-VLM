#!/bin/bash

# =============================================================================
# Fine-tune Qwen3-VL-32B on SPoRC - Conservative LoRA
#
# Dataset: 894 complete excerpts (sporc_excerpt%100)
# Method: Conservative LoRA - attention ONLY, frozen MLP
# Hardware: 8 GPUs (32B model ~64GB, needs more GPUs than 8B)
# =============================================================================

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}  # 32B needs all 8 GPUs
NNODES=${WORLD_SIZE:-1}

# Model configuration
MODEL_PATH="/home/ubuntu/LLM/qwen3-vl-32b"

# Dataset configuration
DATASET="sporc_excerpt%100"  # Use 100% of 894 excerpts

# Output configuration with date prefix
DATE=$(date +%Y-%m-%d)
RUN_NAME="qwen3vl-32b-sporc-lora"
OUTPUT_DIR="/home/ubuntu/image-to-text/Qwen3-VL/finetuned_models/${DATE}_${RUN_NAME}"
LOG_FILE="${OUTPUT_DIR}/training.log"

# Training hyperparameters - Updated based on Gemini review (11/22/25)
# Increased LR and epochs since 168 steps was too conservative
LR=2e-5                   # Increased from 5e-6 (4x higher)
BATCH_SIZE=1              # Keep at 1 due to model size
GRAD_ACCUM_STEPS=2        # Effective batch = 16 (1 × 2 × 8 GPUs)
NUM_EPOCHS=5              # Increased from 3 to 5 (~280 steps total)

# LoRA configuration - Updated 11/22/25: Added LLM MLP layers per Gemini recommendation
USE_LORA=True
LORA_R=16                 # Smaller rank for 32B (less params needed)
LORA_ALPHA=32             # 2x rank
LORA_DROPOUT=0.05
# Attention: q_proj, k_proj, v_proj, o_proj (contextual routing)
# MLP: gate_proj, up_proj, down_proj (reasoning and style/tone)
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Image resolution - use training code defaults (match inference better)
# Defaults: max_pixels=451584 (28*28*576), min_pixels=12544 (28*28*16)
# Previously used restrictive values (50176/784) to save memory
# Now removed to use defaults and match inference processor behavior
# MAX_PIXELS=50176
# MIN_PIXELS=784

# Training entry point
ENTRY_FILE=qwen-vl-finetune/qwenvl/train/train_qwen.py

# DeepSpeed config (ZeRO-3 for memory efficiency)
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
    --save_strategy epoch \
    --save_total_limit 10 \
    --learning_rate ${LR} \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${RUN_NAME} \
    --report_to none"

# Print configuration
echo "================================================================================"
echo "Qwen3-VL-32B Conservative LoRA Fine-tuning"
echo "================================================================================"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET} (894 samples)"
echo ""
echo "Hardware:"
echo "  GPUs: ${NPROC_PER_NODE} × A100 80GB"
echo "  DeepSpeed: ZeRO-3"
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
echo "  LoRA enabled: ${USE_LORA}"
echo "  LoRA rank (r): ${LORA_R}"
echo "  LoRA alpha: ${LORA_ALPHA}"
echo "  LoRA dropout: ${LORA_DROPOUT}"
echo "  Target modules: ${LORA_TARGET_MODULES}"
echo ""
echo "Model components:"
echo "  Vision encoder: ❌ Frozen"
echo "  Projector (bridge): ❌ Frozen"
echo "  LLM Attention: ✅ LoRA (q,k,v,o_proj)"
echo "  LLM MLP: ✅ LoRA (gate,up,down_proj)"
echo ""
echo "Image resolution:"
echo "  Using training code defaults (max=451584, min=12544)"
echo "  Matches inference processor behavior"
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
echo "✓ Saved: $(basename $0) -> ${OUTPUT_DIR}/"
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
echo "  python inference.py --mode interactive --lora-adapter ../${OUTPUT_DIR} --seed 0"
echo ""
echo "Note: We load base model + LoRA adapter on-the-fly (no merging needed)"
echo "================================================================================"
