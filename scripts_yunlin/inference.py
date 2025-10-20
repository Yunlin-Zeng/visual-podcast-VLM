"""
Main inference script for Qwen3-VL-235B
Supports two modes:
  - interactive: Interactive testing with predefined/custom prompts
  - mass: Batch inference on 200 prompts for training data generation

Usage:
  python inference.py --mode interactive
  python inference.py --mode mass [--start-idx 0]
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from utils import load_model, run_inference


def inference_interactive(model, processor):
    """Interactive mode for testing prompts"""

    # Predefined prompts for quick testing
    prompts = {
        "1": {
            "name": "Simple prompt",
            "text": "Please generate a podcast transcript based on these images."
        },
        "2": {
            "name": "Poetic & cheerful",
            "text": "Please generate a transcript in a poetic style, with a cheerful tone, in podcast format, with length between 600-700 words, and elaborate detail level.\n\nCreate an engaging podcast about the events in these images."
        },
        "3": {
            "name": "Documentary & suspenseful",
            "text": "Please generate a transcript in a documentary style, with a suspenseful tone, in podcast format, with length between 600-700 words, and elaborate detail level.\n\nCreate an engaging podcast dialogue based on the visual narrative."
        },
        "4": {
            "name": "Comedy style",
            "text": "Please generate a transcript in a comedy style, with a humorous tone, in podcast format, with length between 600-700 words, and elaborate detail level.\n\nGenerate a funny podcast transcript for these images."
        },
        "5": {
            "name": "Cinematic & calm",
            "text": "Please generate a transcript in a cinematic style, with a calm tone, in podcast format, with length between 600-700 words, and elaborate detail level.\n\nPlease generate a podcast transcript based on these images."
        }
    }

    print("\n" + "=" * 80)
    print("INTERACTIVE INFERENCE MODE")
    print("=" * 80)
    print("\nAvailable prompts:")
    for key, prompt in prompts.items():
        print(f"  {key}. {prompt['name']}")
    print("  c. Custom prompt")
    print("  q. Quit")

    while True:
        print("\n" + "-" * 80)
        choice = input("\nSelect prompt (1-5, c, or q): ").strip().lower()

        if choice == 'q':
            print("Exiting...")
            break

        if choice == 'c':
            print("\nEnter your custom prompt (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "" and lines:
                    break
                lines.append(line)
            prompt_text = "\n".join(lines)
        elif choice in prompts:
            prompt_text = prompts[choice]["text"]
            print(f"\nUsing: {prompts[choice]['name']}")
        else:
            print("Invalid choice!")
            continue

        # Show prompt
        print("\n" + "=" * 80)
        print("PROMPT:")
        print("=" * 80)
        print(prompt_text)

        # Run inference
        output, timing = run_inference(model, processor, prompt_text, verbose=True)

        if output:
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
                    f.write(f"PROMPT:\n{prompt_text}\n\n")
                    f.write(f"OUTPUT:\n{output}\n")
                print(f"✓ Saved to {filename}")


def inference_mass(model, processor, start_idx=0):
    """Batch inference mode for generating 200 training samples"""

    # Paths
    prompts_file = Path("/home/ubuntu/image-to-text/data/2025-10-16_prompt_templates_tune_qwen3_vl/prompts_200.json")
    output_dir = Path("/home/ubuntu/image-to-text/data/2025-10-16_prompt_templates_tune_qwen3_vl/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    vist_train_path = Path("/home/ubuntu/image-to-text/data/visual_storytelling/sis/train.story-in-sequence.json")
    vist_images_dir = Path("/home/ubuntu/image-to-text/data/visual_storytelling/images/train")

    # Load prompts
    print("=" * 80)
    print("Loading prompts and VIST stories...")
    print("=" * 80)

    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    print(f"✓ Loaded {len(prompts)} prompts from {prompts_file}")

    # Load VIST training data and get first 200 unique stories with 5 images each
    print("✓ Loading VIST training stories...")
    with open(vist_train_path, 'r', encoding='utf-8') as f:
        vist_data = json.load(f)

    # Group annotations by story_id
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

    # Get stories with exactly 5 images, sorted by order
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

    print(f"✓ Selected first {len(valid_stories)} stories from VIST training set")

    # Run inference for all prompts
    results = []

    print("\n" + "=" * 80)
    print(f"Starting mass inference on {len(prompts)} prompts...")
    print(f"Starting from prompt {start_idx + 1}")
    print("=" * 80)

    for i, prompt_data in enumerate(prompts[start_idx:], start=start_idx):
        prompt_id = prompt_data['id']
        full_prompt = prompt_data['full_prompt']
        short_prompt = prompt_data['short_prompt']

        # Get corresponding story images
        story_data = valid_stories[i]
        image_paths = [vist_images_dir / f"{img_id}.jpg" for img_id in story_data['image_ids']]

        print(f"\n{'='*80}")
        print(f"PROMPT {prompt_id}/{len(prompts)}")
        print(f"Style: {prompt_data['style'] if prompt_data['style'] else 'default'}")
        print(f"Tone: {prompt_data['tone'] if prompt_data['tone'] else 'default'}")
        print(f"Story ID: {story_data['story_id']}")
        print(f"{'='*80}")

        try:
            # Run inference with full prompt and specific story images
            output, timing_info = run_inference(
                model,
                processor,
                full_prompt,
                image_paths=image_paths,
                verbose=True
            )

            if output is None:
                print(f"❌ Failed to generate output for prompt {prompt_id}")
                continue

            # Store result
            result = {
                "prompt_id": prompt_id,
                "style": prompt_data['style'],
                "tone": prompt_data['tone'],
                "format": prompt_data['format'],
                "length": prompt_data['length'],
                "detail_level": prompt_data['detail_level'],
                "user_prompt": prompt_data['user_prompt'],
                "full_prompt": full_prompt,
                "short_prompt": short_prompt,  # For fine-tuning input
                "generated_transcript": output,
                "word_count": len(output.split()),
                "timing": timing_info,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)

            print(f"\n✓ Generated {result['word_count']} words")

            # Save progress after each prompt (in case of crashes)
            progress_file = output_dir / f"results_progress_{prompt_id}.json"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Also save individual output
            individual_file = output_dir / f"output_{prompt_id:03d}.json"
            with open(individual_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"❌ Error processing prompt {prompt_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Save final results
    final_file = output_dir / "results_all_200.json"
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("MASS INFERENCE COMPLETE")
    print("=" * 80)
    print(f"✓ Successfully generated {len(results)}/{len(prompts)} transcripts")
    print(f"✓ Results saved to: {final_file}")
    print(f"✓ Individual outputs saved to: {output_dir}/")

    # Print statistics
    if results:
        avg_words = sum(r['word_count'] for r in results) / len(results)
        avg_time = sum(r['timing']['total_time'] for r in results) / len(results)
        total_time = sum(r['timing']['total_time'] for r in results)

        print(f"\nStatistics:")
        print(f"  - Average word count: {avg_words:.1f}")
        print(f"  - Average time per prompt: {avg_time:.1f}s")
        print(f"  - Total generation time: {total_time/60:.1f} minutes")


def main():
    """Main entry point with argument parsing"""

    parser = argparse.ArgumentParser(
        description="Qwen3-VL inference script with interactive and mass modes"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "mass"],
        required=True,
        help="Inference mode: 'interactive' for testing, 'mass' for batch generation"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index for mass inference (for resuming)"
    )

    args = parser.parse_args()

    # Load model once
    model, processor = load_model()

    # Run selected mode
    if args.mode == "interactive":
        inference_interactive(model, processor)
    elif args.mode == "mass":
        inference_mass(model, processor, start_idx=args.start_idx)


if __name__ == "__main__":
    main()
