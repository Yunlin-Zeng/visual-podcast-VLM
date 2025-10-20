"""
Analyze token breakdown for Qwen3-VL inference
Shows how input tokens are distributed between text prompts and images
Uses estimation without loading the full model
"""

import json
from pathlib import Path

def estimate_tokens(text):
    """
    Rough estimation: ~1.3 tokens per word for English text
    More accurate would be to use actual tokenizer, but this gives a good approximation
    """
    words = len(text.split())
    # Qwen tokenizer typically produces 1.2-1.4 tokens per word
    return int(words * 1.3)

def analyze_token_breakdown(output_file):
    """
    Analyze token breakdown from a generated output file

    Args:
        output_file: Path to output JSON file
    """

    # Load the output file
    with open(output_file, 'r') as f:
        data = json.load(f)

    # Get the full prompt text
    full_prompt = data['full_prompt']
    short_prompt = data['short_prompt']

    # Estimate prompt tokens
    num_prompt_tokens = estimate_tokens(full_prompt)
    num_short_prompt_tokens = estimate_tokens(short_prompt)

    print("="*80)
    print("TOKEN BREAKDOWN ANALYSIS")
    print("="*80)

    print(f"\n📄 OUTPUT FILE: {output_file.name}")
    print(f"   Prompt ID: {data['prompt_id']}")
    print(f"   Style: {data['style']}")
    print(f"   Tone: {data['tone']}")

    print(f"\n📝 TEXT PROMPT TOKENS:")
    print(f"   Full prompt: {len(full_prompt)} chars, ~{len(full_prompt.split())} words")
    print(f"   Estimated tokens: ~{num_prompt_tokens} tokens")

    print(f"\n   Short prompt: {len(short_prompt)} chars, ~{len(short_prompt.split())} words")
    print(f"   Estimated tokens: ~{num_short_prompt_tokens} tokens")
    print(f"   Token reduction: ~{num_prompt_tokens - num_short_prompt_tokens} tokens ({(1 - num_short_prompt_tokens/num_prompt_tokens)*100:.1f}% reduction)")

    # Image token estimation
    # For Qwen3-VL with resolution 512×32×32:
    # - 512×32×32 = 524,288 pixels per image
    # - Vision encoder processes in 32×32 patches
    # - Total patches = 524,288 / (32×32) = 512 patches
    # - Each patch typically becomes ~3-4 tokens after projection
    # - So ~512 × 3.5 = 1,792 tokens per image

    image_resolution = "512×32×32"
    pixels_per_image = 512 * 32 * 32  # 524,288 pixels
    patch_size = 32
    patches_per_image = pixels_per_image // (patch_size * patch_size)  # 512 patches
    tokens_per_patch = 3.5  # Approximate (varies by model architecture)
    tokens_per_image = int(patches_per_image * tokens_per_patch)
    num_images = 5
    total_image_tokens = tokens_per_image * num_images

    print(f"\n🖼️  IMAGE TOKENS (ESTIMATED):")
    print(f"   Resolution: {image_resolution} = {pixels_per_image:,} pixels per image")
    print(f"   Patch size: {patch_size}×{patch_size}")
    print(f"   Patches per image: {patches_per_image}")
    print(f"   Tokens per patch: ~{tokens_per_patch}")
    print(f"   Tokens per image: ~{tokens_per_image}")
    print(f"   Number of images: {num_images}")
    print(f"   Total image tokens: ~{total_image_tokens}")

    # Total input tokens
    total_input_tokens = num_prompt_tokens + total_image_tokens

    print(f"\n📊 TOTAL INPUT BREAKDOWN:")
    print(f"   Text prompt tokens: ~{num_prompt_tokens:,} ({num_prompt_tokens/total_input_tokens*100:.1f}%)")
    print(f"   Image tokens:       ~{total_image_tokens:,} ({total_image_tokens/total_input_tokens*100:.1f}%)")
    print(f"   {'─'*40}")
    print(f"   Total input tokens: ~{total_input_tokens:,}")

    print(f"\n💡 NOTE: These are estimates. Actual values may differ by ±10-20% due to:")
    print(f"   • Tokenizer's exact behavior (BPE subword splitting)")
    print(f"   • Special tokens (<image>, chat template markers)")
    print(f"   • Vision encoder's projection layer configuration")

    # Output tokens
    output_tokens = data['timing']['num_tokens']
    word_count = data['word_count']

    print(f"\n📤 OUTPUT TOKENS:")
    print(f"   Generated tokens: {output_tokens:,}")
    print(f"   Word count: {word_count}")
    print(f"   Tokens per word: {output_tokens/word_count:.2f}")

    print(f"\n⏱️  PERFORMANCE:")
    print(f"   Input prep time: {data['timing']['input_prep_time']:.2f}s")
    print(f"   Generation time: {data['timing']['generation_time']:.2f}s")
    print(f"   Total time: {data['timing']['total_time']:.2f}s")
    print(f"   Tokens per second: {data['timing']['tokens_per_sec']:.2f}")

    # After fine-tuning projection
    short_total_input = num_short_prompt_tokens + total_image_tokens

    print(f"\n🎯 AFTER FINE-TUNING (PROJECTED):")
    print(f"   Short prompt tokens: ~{num_short_prompt_tokens:,}")
    print(f"   Image tokens:        ~{total_image_tokens:,}")
    print(f"   {'─'*40}")
    print(f"   Total input tokens:  ~{short_total_input:,}")
    print(f"\n   Input token savings: ~{total_input_tokens - short_total_input:,} tokens")
    print(f"   Reduction: {(1 - short_total_input/total_input_tokens)*100:.1f}%")
    print(f"\n   💰 Cost savings: Reduced input from ~{total_input_tokens:,} → ~{short_total_input:,} tokens")
    print(f"      (For API usage: ~{(1 - short_total_input/total_input_tokens)*100:.1f}% cheaper per request)")

    print("\n" + "="*80)

    return {
        'full_prompt_tokens': num_prompt_tokens,
        'short_prompt_tokens': num_short_prompt_tokens,
        'image_tokens': total_image_tokens,
        'total_input_tokens': total_input_tokens,
        'output_tokens': output_tokens
    }

if __name__ == "__main__":
    import sys

    # Allow specifying output file as argument
    if len(sys.argv) > 1:
        output_file = Path(sys.argv[1])
    else:
        # Default to first output
        output_file = Path("/home/ubuntu/image-to-text/data/2025-10-16_prompt_templates_tune_qwen3_vl/outputs/output_001.json")

    if output_file.exists():
        analyze_token_breakdown(output_file)
    else:
        print(f"❌ Output file not found: {output_file}")
        print(f"Usage: python analyze_tokens_simple.py [path/to/output.json]")
