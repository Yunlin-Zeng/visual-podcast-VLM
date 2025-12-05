#!/bin/bash

# =============================================================================
# Fine-tune Qwen3-VL-235B with LoRA to internalize system prompt
#
# Goal: Train on {short_prompt, generated_transcript} pairs using 199 samples
# Method: LoRA fine-tuning (tune LLM + MLP, freeze vision encoder)
# Hardware: 8× A100 80GB
#
# IMPORTANT: Qwen3-VL-235B is MoE and does NOT support DeepSpeed ZeRO-3
#            Must use ZeRO-2 or no DeepSpeed
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
RUN_NAME="qwen3vl-235b-lora-sporc-excerpt"
OUTPUT_DIR="./finetuned_models/${RUN_NAME}"

# Training hyperparameters
# For 235B model with LoRA:
# - Moderate LR for LoRA (1e-6 to 5e-6)
# - Small batch size per device (1-2)
# - Moderate gradient accumulation for more training steps
LR=2e-6
BATCH_SIZE=1
GRAD_ACCUM_STEPS=2  # Effective batch size = 1 × 2 × 8 GPUs = 16

# Image resolution (same as inference: 512×32×32)
# max_pixels = 512 * 32 * 32 = 524,288 pixels
MAX_PIXELS=50176   # Qwen format: 50176 = 1792 × 28 = 512×32×32 converted
MIN_PIXELS=784     # Minimum resolution

# Training entry point
ENTRY_FILE=qwen-vl-finetune/qwenvl/train/train_qwen.py

# DeepSpeed config (use ZeRO-2, NOT ZeRO-3 for MoE models)
DEEPSPEED_CONFIG=qwen-vl-finetune/scripts/zero2.json

# LoRA configuration
USE_LORA=True
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Training arguments
ARGS="
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
    --num_train_epochs 5 \
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
    --save_total_limit 5 \
    --logging_steps 1 \
    --report_to none \
    --eval_strategy no \
    --gradient_checkpointing True \
    --dataloader_num_workers 4
"

# Print configuration
echo "=============================================================================="
echo "Qwen3-VL-235B LoRA Fine-tuning Configuration"
echo "=============================================================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET} (894 samples)"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Hardware:"
echo "  GPUs: ${NPROC_PER_NODE} × A100 80GB"
echo "  DeepSpeed: Disabled (using device_map=auto instead)"
echo ""
echo "Hyperparameters:"
echo "  Learning rate: ${LR}"
echo "  Batch size per device: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "  Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS)) (single process, no data parallelism)"
echo "  Epochs: 5"
echo "  Total steps: ~$((894 * 5 / (BATCH_SIZE * GRAD_ACCUM_STEPS)))"
echo "  Est. training time: ~100 minutes (10 min/epoch × 10 epochs)"
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
echo "  MLP projector: ❌ Frozen"
echo "  LLM: ✅ Trainable (LoRA adapters only)"
echo ""
echo "Image resolution:"
echo "  Max: ${MAX_PIXELS} pixels"
echo "  Min: ${MIN_PIXELS} pixels"
echo "=============================================================================="
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Launch training
python \
         ${ENTRY_FILE} ${ARGS}

echo ""
echo "=============================================================================="
echo "Training completed!"
echo "LoRA adapter saved to: ${OUTPUT_DIR}"
echo ""
echo "To run inference with LoRA adapter:"
echo "  cd /home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin"
echo "  python inference.py --mode interactive --lora-adapter ../${OUTPUT_DIR} --seed 0"
echo ""
echo "Note: We load base model + LoRA adapter on-the-fly (no merging needed)"
echo "=============================================================================="
