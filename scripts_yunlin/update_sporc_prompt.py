"""
Update all 894 SPoRC samples with the working prompt format
(Simplified version with 800 word target to match actual data distribution)
"""
import json

# Working prompt format (simplified, targeting 800 words)
WORKING_PROMPT = """You are creating a podcast transcript in a Q&A format between two hosts based on the image I uploaded. They narrate a descriptive and detailed story based on the images. They can ask questions. Based on the visual information from these 5 images, create an engaging conversational podcast transcript.

CRITICAL REQUIREMENTS:

1. STRUCTURE (REQUIRED):
- One host guides the talk. Sometimes either host can ask questions.
- Both hosts can provide rich visual descriptions
- Questions should follow the story arc: setting → characters (if any) → action → details → conclusion

2. FORMAT (NON-NEGOTIABLE):
- Do NOT cut the conversation short
- Each answer should be substantial

3. VISUAL DETAILS IN ANSWERS (REQUIRED):
- Include SPECIFIC descriptions: exact colors, patterns, materials, textures

4. NATURAL DIALOGUE (REQUIRED):
- DO NOT mention "images", "photos", "pictures"
- Tell the story naturally through conversation

Please generate a transcript in a conversational style and in podcast format, with length around 800 words.

Generate a two-host conversation about this visual sequence."""

# Load original dataset
print("Loading original dataset...")
with open('data/qwen_training_data_sporc_excerpt.json', 'r') as f:
    original_data = json.load(f)

print(f"Total samples: {len(original_data)}")

# Update all samples with working prompt
updated_data = []
for i, sample in enumerate(original_data):
    num_images = len(sample["image"])
    image_tags = "<image>" * num_images

    updated_sample = {
        "image": sample["image"],
        "conversations": [
            {
                "from": "human",
                "value": f"{image_tags}\n{WORKING_PROMPT}"
            },
            {
                "from": "gpt",
                "value": sample["conversations"][1]["value"]  # Keep original ground truth
            }
        ]
    }
    updated_data.append(updated_sample)

    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{len(original_data)} samples...")

# Save updated dataset
output_file = 'data/qwen_training_data_sporc_excerpt_working_prompt.json'
with open(output_file, 'w') as f:
    json.dump(updated_data, f, indent=2)

print(f"\n✓ Updated dataset saved to: {output_file}")
print(f"  - Total samples: {len(updated_data)}")
print(f"  - All samples now use working prompt (800 word target)")
print(f"\nFirst sample preview:")
print(f"  Images: {len(updated_data[0]['image'])}")
print(f"  Prompt length: {len(updated_data[0]['conversations'][0]['value'])} chars")
print(f"  Ground truth words: {len(updated_data[0]['conversations'][1]['value'].split())}")
