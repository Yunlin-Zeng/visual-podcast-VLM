"""
Fine-tune Qwen3-VL-235B with LoRA to internalize system prompt in model weights

Goal: Train on {short_prompt, generated_transcript} pairs so model learns to produce
      high-quality outputs from short prompts without the detailed system instructions

Dataset: 199 samples from /home/ubuntu/image-to-text/data/2025-10-16_prompt_templates_tune_qwen3_vl/outputs/
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import torch
from datasets import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info


def load_training_data(outputs_dir: str, skip_ids: List[int] = [25]) -> List[Dict]:
    """
    Load all output JSON files and prepare training data

    Args:
        outputs_dir: Directory containing output_XXX.json files
        skip_ids: List of IDs to skip (e.g., failed samples)

    Returns:
        List of training examples with format:
        {
            'story_id': str,
            'images': List[str],  # Paths to 5 images
            'short_prompt': str,
            'target_output': str,
            'style': str,
            'tone': str
        }
    """
    outputs_path = Path(outputs_dir)
    training_data = []

    # Load VIST data to get image paths
    vist_train_path = Path("/home/ubuntu/image-to-text/data/visual_storytelling/sis/train.story-in-sequence.json")
    vist_images_dir = Path("/home/ubuntu/image-to-text/data/visual_storytelling/images/train")

    print(f"Loading VIST data from {vist_train_path}...")
    with open(vist_train_path, 'r') as f:
        vist_data = json.load(f)

    # Group by story_id to get image sequences
    from collections import defaultdict
    stories = defaultdict(list)
    for annotation_group in vist_data['annotations']:
        for ann in annotation_group:
            story_id = ann['story_id']
            photo_id = ann['photo_flickr_id']
            order = ann['worker_arranged_photo_order']
            stories[story_id].append({
                'photo_id': photo_id,
                'order': order
            })

    # Get valid stories with 5 images
    valid_stories = []
    for story_id, photos in stories.items():
        if len(photos) == 5:
            photos_sorted = sorted(photos, key=lambda x: x['order'])
            image_ids = [p['photo_id'] for p in photos_sorted]
            valid_stories.append({
                'story_id': story_id,
                'image_ids': image_ids
            })
        if len(valid_stories) >= 200:
            break

    print(f"Found {len(valid_stories)} valid stories from VIST")

    # Load each output file
    for i in range(1, 201):
        if i in skip_ids:
            print(f"Skipping output_{i:03d}.json (in skip list)")
            continue

        output_file = outputs_path / f"output_{i:03d}.json"

        if not output_file.exists():
            print(f"Warning: {output_file} not found, skipping")
            continue

        with open(output_file, 'r') as f:
            data = json.load(f)

        # Get corresponding story images (i-1 because output_001 uses story index 0)
        story_data = valid_stories[i - 1]
        image_paths = [str(vist_images_dir / f"{img_id}.jpg") for img_id in story_data['image_ids']]

        # Verify all images exist
        if not all(Path(p).exists() for p in image_paths):
            print(f"Warning: Missing images for output_{i:03d}.json, skipping")
            continue

        training_data.append({
            'story_id': story_data['story_id'],
            'images': image_paths,
            'short_prompt': data['short_prompt'],
            'target_output': data['generated_transcript'],
            'style': data['style'],
            'tone': data['tone'],
            'word_count': data['word_count']
        })

    print(f"Loaded {len(training_data)} training examples")
    return training_data


def prepare_dataset(training_data: List[Dict], processor) -> Dataset:
    """
    Convert training data to HuggingFace Dataset format

    Format for Qwen3-VL training:
    {
        'messages': [
            {'role': 'user', 'content': [
                {'type': 'image', 'image': 'path1.jpg'},
                {'type': 'image', 'image': 'path2.jpg'},
                ...
                {'type': 'text', 'text': 'short_prompt'}
            ]},
            {'role': 'assistant', 'content': 'target_output'}
        ]
    }
    """
    dataset_entries = []

    for example in training_data:
        # Build user message with 5 images + short prompt
        user_content = []
        for img_path in example['images']:
            user_content.append({'type': 'image', 'image': img_path})
        user_content.append({'type': 'text', 'text': example['short_prompt']})

        # Create messages format
        messages = [
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': example['target_output']}
        ]

        dataset_entries.append({
            'messages': messages,
            'story_id': example['story_id']
        })

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(dataset_entries)
    print(f"Created dataset with {len(dataset)} examples")

    return dataset


def setup_lora_config():
    """
    Configure LoRA for Qwen3-VL-235B

    Target modules for vision-language models:
    - Language model layers: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    - Vision encoder: Usually frozen (not targeted by LoRA)
    """
    lora_config = LoraConfig(
        r=16,  # LoRA rank (16-64 typical, lower = fewer params)
        lora_alpha=32,  # Scaling factor (typically 2*r)
        target_modules=[
            # Language model attention
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # MLP layers
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    return lora_config


def main():
    """Main training function"""

    # ============================================================================
    # Configuration
    # ============================================================================

    MODEL_PATH = "/home/ubuntu/LLM/qwen3-vl-235b"
    OUTPUT_DIR = "/home/ubuntu/image-to-text/Qwen3-VL/finetuned_models/qwen3-vl-235b-lora-short-prompt"
    OUTPUTS_DIR = "/home/ubuntu/image-to-text/data/2025-10-16_prompt_templates_tune_qwen3_vl/outputs"

    # Training hyperparameters
    BATCH_SIZE = 1  # Per device, 235B is huge
    GRADIENT_ACCUM_STEPS = 4  # Effective batch size = 1 * 4 * 8 GPUs = 32
    LEARNING_RATE = 2e-4  # Standard for LoRA
    NUM_EPOCHS = 3  # Start with 3, can adjust
    WARMUP_STEPS = 10
    SAVE_STEPS = 20
    EVAL_STEPS = 20
    LOGGING_STEPS = 5

    # ============================================================================
    # Load Model and Processor
    # ============================================================================

    print("="*80)
    print("Loading Qwen3-VL-235B model and processor...")
    print("="*80)

    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Set image resolution (same as inference)
    processor.image_processor.size = {
        "longest_edge": 512 * 32 * 32,
        "shortest_edge": 256 * 32 * 32
    }

    # Load model with multi-GPU support
    max_memory = {i: "70GB" for i in range(8)}

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True
    )

    print(f"Model loaded successfully")
    print(f"Model device map: {model.hf_device_map}")

    # ============================================================================
    # Prepare Model for LoRA
    # ============================================================================

    print("\n" + "="*80)
    print("Setting up LoRA configuration...")
    print("="*80)

    lora_config = setup_lora_config()
    print(f"LoRA config: rank={lora_config.r}, alpha={lora_config.lora_alpha}")
    print(f"Target modules: {lora_config.target_modules}")

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ============================================================================
    # Load and Prepare Dataset
    # ============================================================================

    print("\n" + "="*80)
    print("Loading training data...")
    print("="*80)

    training_data = load_training_data(OUTPUTS_DIR, skip_ids=[25])

    # Split into train/eval (90/10 split)
    split_idx = int(len(training_data) * 0.9)
    train_data = training_data[:split_idx]
    eval_data = training_data[split_idx:]

    print(f"Train samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")

    train_dataset = prepare_dataset(train_data, processor)
    eval_dataset = prepare_dataset(eval_data, processor)

    # ============================================================================
    # Setup Training Arguments
    # ============================================================================

    print("\n" + "="*80)
    print("Configuring training arguments...")
    print("="*80)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,  # Keep only 3 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=True,  # Use bfloat16
        dataloader_num_workers=4,
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Save memory
        ddp_find_unused_parameters=False,
        report_to="tensorboard",
        logging_dir=f"{OUTPUT_DIR}/logs"
    )

    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUM_STEPS * 8} (across 8 GPUs)")
    print(f"Total training steps: {len(train_dataset) * NUM_EPOCHS // (BATCH_SIZE * GRADIENT_ACCUM_STEPS * 8)}")

    # ============================================================================
    # Training
    # ============================================================================

    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    print("\nNOTE: This is a placeholder. Full implementation requires:")
    print("1. Custom data collator for vision-language inputs")
    print("2. Proper tokenization with processor.apply_chat_template()")
    print("3. Image preprocessing with process_vision_info()")
    print("4. Loss computation setup")
    print("\nRecommendation: Use TRL's SFTTrainer or HuggingFace's example scripts")
    print("See: https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl")
    print("="*80)

    # TODO: Implement custom trainer or use SFTTrainer from TRL
    # This requires additional dependencies and custom data collation

    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_data_collator,  # Need to implement
        tokenizer=processor.tokenizer,
    )

    trainer.train()

    # Save final model
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    """

    print("\nDataset prepared successfully!")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")
    print(f"\nNext step: Implement training loop with proper data collation")


if __name__ == "__main__":
    main()
