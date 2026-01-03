"""
Generate 200 diverse prompts for podcast transcript generation.
Creates both full prompts (for generation) and short prompts (for fine-tuning).
"""

import json
import random
from pathlib import Path

# Fixed system prompt
SYSTEM_PROMPT = """You are creating a podcast transcript in a Q&A format between two hosts based on the image I uploaded. They narrate a descriptive and detailed story based on the images. They can ask questions. Based on the visual information from these 5 images, create an engaging conversational podcast transcript.

CRITICAL REQUIREMENTS:

1. STRUCTURE (REQUIRED):
   - One host guides the talk. Sometimes either host can ask questions.
   - Both hosts can provide rich visual descriptions
   - Questions should follow the story arc: setting → characters (if any) → action → details → conclusion

   Question pattern example:
   - Q1: "What's happening here? Where does this take place?"
   - Q2: "Tell me about the people involved. What are they wearing?"
   - Q3: "What happens next in the story?"
   - Q4: "Are there any other interesting details?"
   - Q5: "How does this story end?"

2. FORMAT (NON-NEGOTIABLE):
   - Do NOT cut the conversation short
   - Each answer should be substantial

3. VISUAL DETAILS IN ANSWERS (REQUIRED):
   Must include SPECIFIC, RICH visual descriptions:

   ✓ GOOD examples (be this specific):
   - "black jacket with bright yellow racing stripes down the sleeves"
   - "light blue plaid boxer shorts with white cross-hatching pattern"
   - "lace curtains filtering soft morning light"
   - "birthday card with 'HAPPY BIRTHDAY' written in bold red marker"
   - "wooden table with warm honey-colored finish"

   ✗ BAD examples (avoid generic descriptions):
   - "nice jacket" (too vague)
   - "blue boxers" (missing pattern detail)
   - "pretty kitchen" (no specific details)

   Details to include in answers:
   - Exact colors and patterns
   - Specific clothing details and materials
   - Textures and finishes
   - Environmental elements (decorations, furniture, wall items)
   - Spatial relationships
   - Lighting and atmosphere

4. NATURAL DIALOGUE (REQUIRED):
   - DO NOT mention "images", "photos", "pictures"
   - Tell the story naturally
   - Questions should feel curious and engaged
   - Answers should be descriptive and informative

   ✓ GOOD:
   What's happening in this birthday scene? Where does this take place?
   This unfolds in the coziest home kitchen. We've got delicate white lace curtains filtering natural light, walls covered with family photos and postcards, even a heart decoration made from blue handprints.

   ✗ BAD:
   What do you see in the first image?
   In the image, there is a kitchen with curtains.

5. (optional) STORY PROGRESSION THROUGH QUESTIONS:
   - Each answer should build on previous information"""

# Variable options (only style and tone vary)
STYLES = [
    "comedy",
    "poetic",
    "dramatic",
    "documentary",
    "casual",
    "educational",
    "storytelling",
    "conversational",
    "analytical",
    "enthusiastic",
    "laid-back",
    "investigative",
    "whimsical",
    "reflective",
    "upbeat",
    "contemplative",
    "journalistic",
    "narrative-driven",
    "observational",
    "cinematic"
]

TONES = [
    "warm",
    "suspenseful",
    "humorous",
    "serious",
    "nostalgic",
    "energetic",
    "mysterious",
    "cheerful",
    "thoughtful",
    "playful",
    "intimate",
    "curious",
    "lighthearted",
    "dramatic",
    "inspiring",
    "melancholic",
    "excited",
    "calm",
    "engaging",
    "friendly"
]

# User prompt templates - diverse ways to ask for podcast generation
USER_PROMPT_TEMPLATES = [
    "Please generate a podcast transcript based on these images.",
    "Create a podcast conversation about what's happening in these images.",
    "Generate a two-host podcast discussing the story in these images.",
    "Produce a podcast transcript that tells the story from these images.",
    "Create an engaging podcast dialogue based on the visual narrative.",
    "Generate a conversational podcast exploring these images.",
    "Please create a podcast transcript narrating the events shown.",
    "Develop a podcast conversation describing the story in detail.",
    "Create a podcast discussing the visual story unfolding here.",
    "Generate a detailed podcast transcript from these images.",
    "Please produce a podcast conversation about this visual story.",
    "Create a podcast that brings these images to life through dialogue.",
    "Generate a podcast transcript exploring what's depicted here.",
    "Develop a conversational podcast narrating these events.",
    "Create a podcast discussing the narrative shown in the images.",
    "Generate a two-host conversation about this visual sequence.",
    "Please create a podcast exploring the story behind these images.",
    "Produce a podcast transcript that describes the scene in detail.",
    "Create an engaging podcast about the events in these images.",
    "Generate a podcast conversation unpacking this visual story.",
    # More varied templates
    "Tell this story through a podcast format.",
    "Turn these images into a podcast conversation.",
    "Create a podcast episode about what you see here.",
    "Generate a dialogue-based podcast from this visual narrative.",
    "Please narrate this story as a podcast between two hosts.",
    "Develop a podcast conversation that captures this moment.",
    "Create a podcast exploring the details in these images.",
    "Generate a conversational podcast about this sequence of events.",
    "Turn this visual story into an engaging podcast transcript.",
    "Create a podcast that discusses every detail shown here.",
]

def generate_variable_section(style, tone):
    """Generate the variable section as a natural sentence."""
    parts = []

    if style:
        parts.append(f"in a {style} style")
    if tone:
        parts.append(f"with a {tone} tone")

    # Always include these fixed parameters
    parts.append("in podcast format")
    parts.append("with length between 600-700 words (do NOT exceed 700 words)")
    parts.append("and elaborate detail level")

    if parts:
        return "Please generate a transcript " + ", ".join(parts) + "."
    return ""

def generate_prompts(num_prompts=200):
    """Generate diverse prompts with both full and short versions."""
    prompts = []

    for i in range(num_prompts):
        # Randomly select style and tone (allow empty for variety)
        style = random.choice(STYLES + [""])  # 20/21 chance of having style
        tone = random.choice(TONES + [""])    # 20/21 chance of having tone

        # Select a user prompt template
        user_prompt = random.choice(USER_PROMPT_TEMPLATES)

        # Generate variable section
        variable_section = generate_variable_section(style, tone)

        # Create short prompt (variable + user)
        short_prompt = f"{variable_section}\n\n{user_prompt}"

        # Create full prompt (system + variable + user)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{variable_section}\n\n{user_prompt}"

        prompts.append({
            "id": i + 1,
            "style": style,
            "tone": tone,
            "format": "podcast",
            "length": "600-700 words",
            "detail_level": "elaborate",
            "user_prompt": user_prompt,
            "variable_section": variable_section,
            "short_prompt": short_prompt,
            "full_prompt": full_prompt
        })

    return prompts

def main():
    # Generate 200 prompts
    print("Generating 200 diverse prompts...")
    prompts = generate_prompts(200)

    # Create output directory
    output_dir = Path("/home/ubuntu/image-to-text/data/2025-10-16_prompt_templates_tune_qwen3_vl")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON file
    output_file = output_dir / "prompts_200.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(prompts)} prompts to {output_file}")

    # Also save a separate file with just short prompts for easy inspection
    short_prompts_file = output_dir / "short_prompts_200.txt"
    with open(short_prompts_file, 'w', encoding='utf-8') as f:
        for p in prompts:
            f.write(f"=== Prompt {p['id']} ===\n")
            f.write(f"Style: {p['style'] if p['style'] else 'default'}\n")
            f.write(f"Tone: {p['tone'] if p['tone'] else 'default'}\n")
            f.write(f"\n{p['short_prompt']}\n")
            f.write("\n" + "="*80 + "\n\n")

    print(f"✓ Saved short prompts to {short_prompts_file}")

    # Print statistics
    styles_used = [p['style'] for p in prompts if p['style']]
    tones_used = [p['tone'] for p in prompts if p['tone']]

    print(f"\nStatistics:")
    print(f"  - Total prompts: {len(prompts)}")
    print(f"  - Prompts with style: {len(styles_used)}")
    print(f"  - Prompts with tone: {len(tones_used)}")
    print(f"  - Unique styles used: {len(set(styles_used))}")
    print(f"  - Unique tones used: {len(set(tones_used))}")
    print(f"  - Unique user prompts: {len(set(p['user_prompt'] for p in prompts))}")

    # Show a few examples
    print(f"\n=== Example Prompts ===")
    for i in [0, 50, 100, 150, 199]:
        p = prompts[i]
        print(f"\n--- Prompt {p['id']} ---")
        print(f"Style: {p['style'] if p['style'] else 'default'}")
        print(f"Tone: {p['tone'] if p['tone'] else 'default'}")
        print(f"\nShort prompt:\n{p['short_prompt']}")

if __name__ == "__main__":
    main()
