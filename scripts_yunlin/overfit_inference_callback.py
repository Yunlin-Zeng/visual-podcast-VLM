"""
Custom callback to run inference during training and save outputs every N epochs
"""
import os
import json
import torch
from transformers import TrainerCallback
from pathlib import Path


class OverfitInferenceCallback(TrainerCallback):
    """
    Callback that runs inference on the training sample every N epochs
    and saves the output to track learning progress
    """

    def __init__(self, inference_frequency=5, data_path=None, output_dir=None):
        """
        Args:
            inference_frequency: Run inference every N epochs (default: 5)
            data_path: Path to the training data JSON file
            output_dir: Directory to save inference outputs
        """
        self.inference_frequency = inference_frequency
        self.data_path = data_path
        self.output_dir = output_dir
        self.inference_outputs_dir = None

    def on_epoch_end(self, args, state, control, model, **kwargs):
        """Run inference at the end of specified epochs"""

        # Only run on main process
        if state.is_world_process_zero:
            current_epoch = int(state.epoch)

            # Check if we should run inference this epoch
            if current_epoch % self.inference_frequency == 0 or current_epoch == 1:

                # Create inference outputs directory if needed
                if self.inference_outputs_dir is None:
                    self.inference_outputs_dir = os.path.join(
                        args.output_dir, "inference_progress"
                    )
                    os.makedirs(self.inference_outputs_dir, exist_ok=True)

                print(f"\n{'='*80}")
                print(f"Running inference at epoch {current_epoch}...")
                print(f"{'='*80}")

                try:
                    # Load training data to get the sample
                    with open(self.data_path, 'r') as f:
                        data = json.load(f)

                    sample = data[0]

                    # Resolve image paths (they might be relative to the data file location)
                    data_dir = Path(self.data_path).parent
                    image_paths = []
                    for img in sample["image"]:
                        img_path = Path(img)
                        if not img_path.is_absolute():
                            # Try relative to data file directory first
                            resolved_path = (data_dir / img_path).resolve()
                            if not resolved_path.exists():
                                # Try relative to current working directory
                                resolved_path = img_path.resolve()
                        else:
                            resolved_path = img_path
                        image_paths.append(resolved_path)

                    prompt_text = sample["conversations"][0]["value"].replace(
                        "<image>" * len(sample["image"]), ""
                    ).strip()

                    # Get processor from kwargs
                    processor = kwargs.get('tokenizer')  # In HF, tokenizer is the processor

                    # Create messages
                    messages = [{
                        "role": "user",
                        "content": [
                            *[{"type": "image", "image": str(p)} for p in image_paths],
                            {"type": "text", "text": prompt_text}
                        ]
                    }]

                    # Prepare inputs
                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt"
                    )
                    inputs = inputs.to(model.device)

                    # Run inference with greedy decoding
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            do_sample=False  # Greedy for consistency
                        )

                    # Decode output
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]

                    # Save output
                    output_file = os.path.join(
                        self.inference_outputs_dir,
                        f"epoch_{current_epoch:03d}_output.txt"
                    )

                    with open(output_file, 'w') as f:
                        f.write(f"Epoch: {current_epoch}\n")
                        f.write(f"Training loss: {state.log_history[-1].get('loss', 'N/A')}\n")
                        f.write(f"Word count: {len(output_text.split())}\n")
                        f.write(f"\n{'='*80}\n")
                        f.write(f"GENERATED OUTPUT:\n")
                        f.write(f"{'='*80}\n\n")
                        f.write(output_text)

                    print(f"✓ Saved inference output to: {output_file}")
                    print(f"  - Generated {len(output_text.split())} words")
                    print(f"  - Training loss: {state.log_history[-1].get('loss', 'N/A')}")
                    print(f"{'='*80}\n")

                except Exception as e:
                    print(f"❌ Error during inference callback: {e}")
                    import traceback
                    traceback.print_exc()

        return control
