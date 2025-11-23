"""
Interactive inference using official HuggingFace approach (no custom processor settings)
This uses the exact loading method from HF docs to test if custom settings cause issues
"""

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from pathlib import Path
import torch
import argparse
import time


def run_inference_official(model, processor, prompt_text, image_paths, seed=None, use_sampling=False, verbose=True):
    """
    Run inference using official HuggingFace approach

    Args:
        model: Loaded model
        processor: Loaded processor
        prompt_text: Text prompt
        image_paths: List of image paths
        seed: Random seed for reproducibility
        use_sampling: If True, use temperature+top_p; if False, use greedy
        verbose: Print timing info

    Returns:
        str: Generated text
    """
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if verbose:
            print(f"✓ Random seed set to: {seed}")

    start_time = time.time()

    # Create messages with all 5 images
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_paths[0])},
                {"type": "image", "image": str(image_paths[1])},
                {"type": "image", "image": str(image_paths[2])},
                {"type": "image", "image": str(image_paths[3])},
                {"type": "image", "image": str(image_paths[4])},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    # Preparation for inference (official HF approach)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    prep_time = time.time()
    if verbose:
        print(f"\n✓ Using {len(image_paths)} images")
        print(f"✓ Input tokens: {inputs['input_ids'].shape[1]}")
        print(f"✓ Input preparation time: {prep_time - start_time:.2f}s")
        print("\n🚀 Generating...")

    # Inference: Generation of the output
    if use_sampling:
        if verbose:
            print("   Method: Sampling (temperature=0.7, top_p=0.8)")
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.8
        )
    else:
        if verbose:
            print("   Method: Greedy decoding")
        generated_ids = model.generate(**inputs, max_new_tokens=1024)

    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    gen_time = time.time()
    if verbose:
        print(f"\n⏱️  Timing:")
        print(f"   - Input preparation: {prep_time - start_time:.2f}s")
        print(f"   - Generation: {gen_time - prep_time:.2f}s")
        print(f"   - Total: {gen_time - start_time:.2f}s")
        print(f"   - Generated {len(generated_ids_trimmed[0])} tokens at {len(generated_ids_trimmed[0])/(gen_time - prep_time):.2f} tokens/s")

    return output_text[0]


def inference_interactive(model, processor, seed=None):
    """Interactive mode for testing prompts"""

    print("\n" + "=" * 80)
    print("INTERACTIVE INFERENCE MODE (Official HuggingFace Approach)")
    print("=" * 80)
    print("\nGeneration modes:")
    print("  g. Greedy decoding (recommended for testing)")
    print("  s. Sampling (temperature=0.7, top_p=0.8)")
    print("  q. Quit")

    while True:
        print("\n" + "=" * 80)
        print("NEW INFERENCE SESSION")
        print("=" * 80)

        # Step 1: Choose generation mode
        mode_choice = input("\nSelect generation mode (g/s/q): ").strip().lower()

        if mode_choice == 'q':
            print("Exiting...")
            break

        if mode_choice not in ['g', 's']:
            print("Invalid choice! Please select 'g', 's', or 'q'")
            continue

        use_sampling = (mode_choice == 's')

        # Step 2: Get image paths
        print("\nEnter image paths (one per line)")
        print("Type 'done' when finished, or press Enter on first image to use default test images")
        print("Examples:")
        print("  /home/ubuntu/image-to-text/data/visual_storytelling/test_samples/sample_06_story_45535/image_1.jpg")
        print()

        image_paths = []

        first_input = input("Image 1 (or press Enter for default): ").strip()
        if first_input == "":
            # Use default images
            print("✓ Using default test images (sample_06_story_45535)")
            image_paths = [
                Path("/home/ubuntu/image-to-text/data/visual_storytelling/test_samples/sample_06_story_45535/image_1.jpg"),
                Path("/home/ubuntu/image-to-text/data/visual_storytelling/test_samples/sample_06_story_45535/image_2.jpg"),
                Path("/home/ubuntu/image-to-text/data/visual_storytelling/test_samples/sample_06_story_45535/image_3.jpg"),
                Path("/home/ubuntu/image-to-text/data/visual_storytelling/test_samples/sample_06_story_45535/image_4.jpg"),
                Path("/home/ubuntu/image-to-text/data/visual_storytelling/test_samples/sample_06_story_45535/image_5.jpg"),
            ]
        else:
            # Got first image, continue collecting more
            path1 = Path(first_input)
            if not path1.exists():
                print(f"❌ File not found: {path1}")
                continue
            image_paths.append(path1)
            print(f"  ✓ Added image 1")

            # Get remaining images
            img_num = 2
            while True:
                path_input = input(f"Image {img_num} (or type 'done' to finish): ").strip()

                if path_input.lower() == 'done':
                    break

                if path_input.lower() == 'q':
                    print("Exiting...")
                    return

                if path_input == "":
                    print("  ⚠️  Empty input. Type 'done' when finished, or enter a path")
                    continue

                path = Path(path_input)
                if path.exists():
                    image_paths.append(path)
                    print(f"  ✓ Added image {img_num}")
                    img_num += 1
                else:
                    print(f"  ❌ File not found: {path}")
                    print(f"  Please enter a valid path or type 'done' to finish")

            print(f"\n✓ Using {len(image_paths)} custom images")

        if len(image_paths) != 5:
            print(f"⚠️  Warning: Expected 5 images, got {len(image_paths)}")
            cont = input("Continue anyway? (y/n): ").strip().lower()
            if cont != 'y':
                continue

        # Step 3: Get prompt
        print("\n" + "-" * 80)
        print("\nEnter your prompt (press Enter three times when done):")
        lines = []
        empty_count = 0
        while True:
            line = input()
            if line == "":
                empty_count += 1
                if empty_count >= 3:
                    break
                lines.append(line)  # Keep empty lines as part of prompt
            else:
                empty_count = 0
                lines.append(line)

        # Remove trailing empty lines
        while lines and lines[-1] == "":
            lines.pop()

        prompt_text = "\n".join(lines)

        if not prompt_text.strip():
            print("❌ Empty prompt! Please try again.")
            continue

        # Show prompt
        print("\n" + "=" * 80)
        print("PROMPT:")
        print("=" * 80)
        print(prompt_text)

        # Run inference
        try:
            output = run_inference_official(
                model, processor, prompt_text, image_paths,
                seed=seed, use_sampling=use_sampling, verbose=True
            )

            print("\n" + "=" * 80)
            print("GENERATED OUTPUT:")
            print("=" * 80)
            print(output)
            print("\n" + "=" * 80)
            print(f"✓ Generated {len(output.split())} words")
            print("=" * 80)

            # Ask if want to save
            save = input("\nSave output to file? (y/n): ").strip().lower()
            if save == 'y':
                filename = input("Enter filename (e.g., output_1.txt): ").strip()
                if not filename:
                    filename = "output.txt"
                with open(filename, 'w') as f:
                    f.write("IMAGES:\n")
                    for i, p in enumerate(image_paths, 1):
                        f.write(f"  {i}. {p}\n")
                    f.write("\n")
                    f.write(f"PROMPT:\n{prompt_text}\n\n")
                    f.write(f"OUTPUT:\n{output}\n")
                print(f"✓ Saved to {filename}")

        except Exception as e:
            print(f"❌ Error during inference: {e}")
            import traceback
            traceback.print_exc()

        # Ask if continue
        cont = input("\nRun another inference? (y/n): ").strip().lower()
        if cont != 'y':
            print("Exiting...")
            break


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="Official HuggingFace inference (no custom processor settings)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive"],
        default="interactive",
        help="Inference mode (currently only interactive)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/ubuntu/LLM/qwen3-vl-8b",
        help="Path to model weights"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Load model using official HuggingFace approach (no custom settings)
    print("=" * 80)
    print("Loading model using official HuggingFace approach...")
    print("=" * 80)
    print(f"Model: {args.model_path}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(args.model_path)

    print("\n✓ Model loaded successfully!")
    print(f"✓ dtype: {model.dtype}")
    print(f"✓ Using DEFAULT processor settings (no custom modifications)")

    if args.seed is not None:
        print(f"✓ Random seed: {args.seed}")

    # Run interactive mode
    inference_interactive(model, processor, seed=args.seed)


if __name__ == "__main__":
    main()
