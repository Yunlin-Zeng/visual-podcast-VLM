"""
Training script for Qwen3-VL with precomputed vision embeddings.

This is a simplified first version that demonstrates loading precomputed embeddings.
The vision components are still loaded for now, but we use precomputed embeddings
to save computation time. Future iterations will remove vision components entirely.
"""

import copy
import logging
import os
import pathlib
import torch
import transformers
import sys
from pathlib import Path

# Add project root to path so we can import qwenvl modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor,
    Trainer
)
from peft import LoraConfig, get_peft_model

from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
from qwenvl.data.data_processor import rank0_print
from qwenvl.data.data_processor_precomputed import create_precomputed_dataset, PrecomputedDataCollator
from qwenvl.train.trainer import replace_qwen2_vl_attention_class

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

local_rank = None


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    """Set which parts of the model are trainable."""
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            if "merger" not in n:
                p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            if "merger" not in n:
                p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def apply_lora(model_args, model):
    """Apply LoRA to the language model"""
    if not model_args.use_lora:
        return model

    rank0_print("Applying LoRA configuration...")

    # Parse target modules
    target_modules = model_args.lora_target_modules.split(",")

    # LoRA configuration
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["visual.merger"] if model_args.tune_mm_mlp else None,
    )

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Manually enable MLP merger if needed
    if model_args.tune_mm_mlp:
        for n, p in model.base_model.model.visual.merger.named_parameters():
            p.requires_grad = True

    rank0_print("LoRA applied successfully")
    return model


def delete_vision_components(model):
    """
    Delete vision encoder and MLP components to free memory.

    This is a placeholder for future implementation. Currently kept simple to
    ensure the training works first.
    """
    rank0_print("=" * 80)
    rank0_print("NOTE: Vision components are still loaded in this version.")
    rank0_print("Future versions will implement full deletion to save memory.")
    rank0_print("=" * 80)

    # TODO: Implement vision component deletion
    # The challenge is that the model's forward expects these components.
    # Options:
    # 1. Create a custom forward method
    # 2. Replace vision forward with dummy that uses cached embeddings
    # 3. Load only LLM weights from checkpoint

    pass


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Check if using precomputed embeddings
    if not data_args.use_precomputed_embeddings:
        raise ValueError(
            "This script requires --use_precomputed_embeddings True "
            "and --precomputed_embeddings_dir <path>"
        )

    if not data_args.precomputed_embeddings_dir:
        raise ValueError("Must specify --precomputed_embeddings_dir when using precomputed embeddings")

    rank0_print("=" * 80)
    rank0_print("Training with PRECOMPUTED VISION EMBEDDINGS")
    rank0_print(f"Precomputed embeddings directory: {data_args.precomputed_embeddings_dir}")
    rank0_print("=" * 80)

    # Prepare quantization config if load_in_4bit is enabled
    quantization_config = None
    if model_args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,
        )
        print(f"Using 4-bit quantization: {model_args.bnb_4bit_quant_type}, compute_dtype={model_args.bnb_4bit_compute_dtype}")

    # Check config.json to detect MoE models
    import json
    config_path = Path(model_args.model_name_or_path) / "config.json"
    is_moe = False
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            model_type = config.get("model_type", "")
            is_moe = "moe" in model_type.lower()
            print(f"Detected model_type from config: {model_type} (MoE: {is_moe})")

    # Load model WITHOUT vision components to save memory
    # Strategy: Use LLM-only checkpoint with filtered weight index
    # The checkpoint at model_name_or_path should have vision weights excluded from index
    rank0_print("=" * 80)
    rank0_print("PHASE 2: Loading model WITHOUT vision encoder and MLP")
    rank0_print("Checkpoint must have filtered weight index (vision weights excluded)")
    rank0_print("=" * 80)

    # Load model
    if "qwen3" in model_args.model_name_or_path.lower() and is_moe:
        rank0_print("Loading Qwen3VL-MoE from LLM-only checkpoint...")
        rank0_print("Expected: Vision weights NOT in weight index → won't be loaded")

        # Load model in float32 first (stays on CPU), then delete vision, then convert
        rank0_print("Loading model on CPU (no dtype conversion yet)...")
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.float32,  # Keep on CPU in FP32 temporarily
            quantization_config=quantization_config,
            # NO device_map - stays on CPU until we manually move it
        )
        data_args.model_type = "qwen3vl"

        rank0_print("✓ Model loaded from filtered checkpoint (CPU, FP32)")

        # CRITICAL: Delete vision components NOW while still on CPU
        # Vision components were randomly initialized (not in checkpoint)
        # Deleting them prevents wasting GPU memory
        rank0_print("Deleting vision encoder and MLP components...")
        if hasattr(model, 'model') and hasattr(model.model, 'visual'):
            # Count parameters before deletion
            visual_params = sum(p.numel() for p in model.model.visual.parameters())
            visual_m_params = visual_params / 1_000_000

            # Delete the visual component
            del model.model.visual
            import gc
            gc.collect()

            rank0_print(f"✓ Deleted vision components ({visual_m_params:.1f}M parameters)")
        else:
            rank0_print("⚠ Warning: model.visual not found, cannot delete")

        # Now convert to bfloat16 if requested (still on CPU)
        if training_args.bf16 and not model_args.load_in_4bit:
            rank0_print("Converting model to bfloat16...")
            model = model.to(torch.bfloat16)
            rank0_print("✓ Converted to bfloat16")

        rank0_print("=" * 80)
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 and not model_args.load_in_4bit else None),
            quantization_config=quantization_config,
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 and not model_args.load_in_4bit else None),
            quantization_config=quantization_config,
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 and not model_args.load_in_4bit else None),
            quantization_config=quantization_config,
        )
        data_args.model_type = "qwen2vl"

    print(f'the initialized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')

    # Load processor (needed for tokenization)
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Apply LoRA if enabled, otherwise use standard fine-tuning
    if model_args.use_lora:
        model = apply_lora(model_args, model)
    else:
        set_model(model_args, model)

    # Print trainable parameters
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()
        else:
            try:
                model.visual.print_trainable_parameters()
                model.model.print_trainable_parameters()
            except:
                pass

    # Delete vision components to save memory (TODO: implement fully)
    delete_vision_components(model)

    # Create dataset with precomputed embeddings
    rank0_print("Loading precomputed embeddings dataset...")
    train_dataset = create_precomputed_dataset(
        processor,
        data_args,
        data_args.precomputed_embeddings_dir
    )
    data_collator = PrecomputedDataCollator(processor.tokenizer)

    rank0_print(f"Dataset loaded: {len(train_dataset)} samples")

    # Create data module
    data_module = dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )

    # Setup trainer
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module
    )

    # Train
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
