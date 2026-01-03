#!/bin/bash

# =============================================================================
# Overfit Test: Train Qwen3-VL-8B on Single Sample
#
# Goal: Verify model CAN learn by overfitting on 1 sample with working prompt
# Method: LoRA fine-tuning on 1 sample for 100 epochs until loss → 0
# Dataset: Sample 4 (ep1017_ex0, 640 words, balanced speakers: S1=6 S2=6)
# Expected: If model is capable, loss should approach 0 and generate exact output
# =============================================================================

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}  # Number of GPUs

# Model configuration
MODEL_PATH="/home/ubuntu/LLM/qwen3-vl-8b"

# Dataset configuration - SINGLE SAMPLE
DATASET="sporc_overfit_single%100"  # 1 sample

# Output configuration
RUN_NAME="qwen3vl-8b-overfit-test"
OUTPUT_DIR="./finetuned_models/${RUN_NAME}"
LOG_FILE="${OUTPUT_DIR}/overfit_training.log"

# Training hyperparameters for OVERFITTING
# Goal: Make loss → 0, so we use aggressive settings
LR=1e-4                  # High LR for fast overfitting (2x original)
BATCH_SIZE=1             # Single sample
GRAD_ACCUM_STEPS=1       # Effective batch size = 1
NUM_EPOCHS=100           # Many epochs to fully overfit

# Image resolution (same as inference: 512×32×32)
MAX_PIXELS=50176         # Qwen format: 512×32×32 converted
MIN_PIXELS=784

# Training entry point
ENTRY_FILE=qwen-vl-finetune/qwenvl/train/train_qwen.py

# DeepSpeed config
DEEPSPEED_CONFIG=qwen-vl-finetune/scripts/zero3.json

# LoRA configuration (same as original)
USE_LORA=True
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Training arguments
ARGS="
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
    --run_name ${RUN_NAME} \
    --num_train_epochs ${NUM_EPOCHS} \
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
    --model_max_length 4096 \
    --save_strategy epoch \
    --save_steps 10 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --report_to none \
    --eval_strategy no \
    --gradient_checkpointing True \
    --dataloader_num_workers 4
"

# Print configuration
echo "=============================================================================="
echo "Qwen3-VL-8B Overfit Test - Single Sample"
echo "=============================================================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET} (1 sample: ep1017_ex0, 640 words)"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Training Configuration:"
echo "  Learning rate: ${LR} (high for fast overfitting)"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "  Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE))"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Total steps: ${NUM_EPOCHS}"
echo "  Est. training time: ~1-2 hours (4 GPUs)"
echo ""
echo "LoRA Configuration:"
echo "  LoRA rank (r): ${LORA_R}"
echo "  LoRA alpha: ${LORA_ALPHA}"
echo "  Target modules: ${LORA_TARGET_MODULES}"
echo ""
echo "Expected Result:"
echo "  - Loss should decrease to near 0 (<0.1)"
echo "  - Model should memorize the exact 640-word dialogue"
echo "  - If this works, model IS capable of learning this format"
echo "  - If this fails, model has fundamental limitation"
echo "=============================================================================="
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Create output directory and launch training
mkdir -p ${OUTPUT_DIR}

# Launch training (output redirected to log file in output directory)
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${ENTRY_FILE} ${ARGS} 2>&1 | tee ${LOG_FILE}

echo ""
echo "=============================================================================="
echo "Overfit test training completed!"
echo "=============================================================================="
echo "Next steps:"
echo "  1. Check final loss - should be < 0.1 if overfitting worked"
echo "  2. Run inference with this LoRA adapter on same images"
echo "  3. Compare output to expected 640-word dialogue (ep1017_ex0)"
echo ""
echo "To test inference:"
echo "  cd /home/ubuntu/image-to-text/Qwen3-VL/scripts_yunlin"
echo "  python inference.py --mode interactive \\"
echo "    --model-path /home/ubuntu/LLM/qwen3-vl-8b \\"
echo "    --lora-adapter ../${OUTPUT_DIR}/checkpoint-100 \\"
echo "    --seed 0"
echo "=============================================================================="
