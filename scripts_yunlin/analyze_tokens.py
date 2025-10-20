"""
Analyze token breakdown for Qwen3-VL inference
Shows how input tokens are distributed between text prompts and images
"""

import json
from pathlib import Path
from transformers import AutoProcessor

def analyze_token_breakdown(output_file, tokenizer_path="/home/ubuntu/LLM/qwen3-vl-235b"):
    """
    Analyze token breakdown from a generated output file

    Args:
        output_file: Path to output JSON file
        tokenizer_path: Path to Qwen3-VL tokenizer
    """

    # Load the output file
    with open(output_file, 'r') as f:
        data = json.load(f)

    # Load processor (includes tokenizer)
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    # Get the full prompt text
    full_prompt = data['full_prompt']

    # Tokenize the text prompt
    prompt_tokens = tokenizer.encode(full_prompt, add_special_tokens=True)
    num_prompt_tokens = len(prompt_tokens)

    print("="*80)
    print("TOKEN BREAKDOWN ANALYSIS")
    print("="*80)

    print(f"\n📄 OUTPUT FILE: {output_file.name}")
    print(f"   Prompt ID: {data['prompt_id']}")
    print(f"   Style: {data['style']}")
    print(f"   Tone: {data['tone']}")

    print(f"\n📝 TEXT PROMPT TOKENS:")
    print(f"   Full prompt length: {len(full_prompt)} characters")
    print(f"   Full prompt tokens: {num_prompt_tokens} tokens")

    # Calculate short prompt tokens for comparison
    short_prompt = data['short_prompt']
    short_prompt_tokens = tokenizer.encode(short_prompt, add_special_tokens=True)
    num_short_prompt_tokens = len(short_prompt_tokens)

    print(f"\n   Short prompt length: {len(short_prompt)} characters")
    print(f"   Short prompt tokens: {num_short_prompt_tokens} tokens")
    print(f"   Token reduction after fine-tuning: {num_prompt_tokens - num_short_prompt_tokens} tokens ({(1 - num_short_prompt_tokens/num_prompt_tokens)*100:.1f}% reduction)")

    # Image token estimation
    # For Qwen3-VL: Image resolution 512×32×32 = 524,288 pixels
    # Rule of thumb: ~3-4 tokens per patch, with patches being 32×32 pixels
    # Total patches = 524,288 / (32*32) = 512 patches per image
    # Each patch typically becomes 2-4 tokens (vision encoder + projection)

    image_resolution = "512×32×32"
    pixels_per_image = 512 * 32 * 32  # 524,288 pixels
    patches_per_image = 512  # Approximate
    tokens_per_patch = 3.5  # Approximate (varies by model)
    estimated_tokens_per_image = int(patches_per_image * tokens_per_patch)
    num_images = 5
    estimated_image_tokens = estimated_tokens_per_image * num_images

    print(f"\n🖼️  IMAGE TOKENS (ESTIMATED):")
    print(f"   Image resolution: {image_resolution} = {pixels_per_image:,} pixels per image")
    print(f"   Patches per image: ~{patches_per_image}")
    print(f"   Tokens per patch: ~{tokens_per_patch}")
    print(f"   Tokens per image: ~{estimated_tokens_per_image}")
    print(f"   Number of images: {num_images}")
    print(f"   Total image tokens: ~{estimated_image_tokens}")

    # Total input tokens
    estimated_total_input = num_prompt_tokens + estimated_image_tokens

    print(f"\n📊 TOTAL INPUT BREAKDOWN:")
    print(f"   Text prompt tokens: {num_prompt_tokens} ({num_prompt_tokens/estimated_total_input*100:.1f}%)")
    print(f"   Image tokens (est): {estimated_image_tokens} ({estimated_image_tokens/estimated_total_input*100:.1f}%)")
    print(f"   Total input tokens: ~{estimated_total_input}")

    # Compare with actual if available (from model logs)
    print(f"\n💡 NOTE: The actual total might be slightly different due to:")
    print(f"   - Special tokens for image boundaries")
    print(f"   - Chat template formatting tokens")
    print(f"   - Vision encoder's exact tokenization")

    # Output tokens
    output_tokens = data['timing']['num_tokens']
    word_count = data['word_count']

    print(f"\n📤 OUTPUT TOKENS:")
    print(f"   Generated tokens: {output_tokens}")
    print(f"   Word count: {word_count}")
    print(f"   Tokens per word: {output_tokens/word_count:.2f}")

    print(f"\n⏱️  PERFORMANCE:")
    print(f"   Generation time: {data['timing']['generation_time']:.2f}s")
    print(f"   Tokens per second: {data['timing']['tokens_per_sec']:.2f}")

    # After fine-tuning projection
    print(f"\n🎯 AFTER FINE-TUNING (PROJECTED):")
    short_total_input = num_short_prompt_tokens + estimated_image_tokens
    print(f"   Short prompt tokens: {num_short_prompt_tokens}")
    print(f"   Image tokens: {estimated_image_tokens}")
    print(f"   Total input tokens: ~{short_total_input}")
    print(f"   Input token reduction: {estimated_total_input - short_total_input} tokens ({(1 - short_total_input/estimated_total_input)*100:.1f}% reduction)")

    print("\n" + "="*80)

    return {
        'full_prompt_tokens': num_prompt_tokens,
        'short_prompt_tokens': num_short_prompt_tokens,
        'estimated_image_tokens': estimated_image_tokens,
        'estimated_total_input': estimated_total_input,
        'output_tokens': output_tokens
    }

if __name__ == "__main__":
    # Analyze first output
    output_file = Path("/home/ubuntu/image-to-text/data/2025-10-16_prompt_templates_tune_qwen3_vl/outputs/output_001.json")

    if output_file.exists():
        analyze_token_breakdown(output_file)
    else:
        print(f"❌ Output file not found: {output_file}")
