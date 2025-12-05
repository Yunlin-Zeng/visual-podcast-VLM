#!/usr/bin/env python3
"""
Update SPoRC dataset with new 235B-aligned prompt (topics/themes focused)
"""

import json

# New SPoRC-aligned prompt
NEW_PROMPT = """<image><image><image><image><image>

Create a podcast transcript where two hosts have a natural conversation about the topics or themes related to these images.

The hosts should:

- Discuss the topics, experiences, or themes that connect to what they see
- Use conversational language
- Have natural back-and-forth exchanges with conversational flow
- Build on each other's points and react naturally
- Mention visual details naturally as they relate to the discussion
- Let the conversation wander and flow organically

The output should be around 800 words. Make it sound like two people casually talking about something interesting, not narrating what's in the images."""

# Load existing dataset
input_file = "/home/ubuntu/image-to-text/Qwen3-VL/data/qwen_training_data_sporc_detailed_prompt.json"
output_file = "/home/ubuntu/image-to-text/Qwen3-VL/data/qwen_training_data_sporc_235b_aligned.json"

print(f"Loading dataset from: {input_file}")
with open(input_file, 'r') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# Update prompts
for sample in data:
    # Update the human message (first in conversations)
    sample['conversations'][0]['value'] = NEW_PROMPT

print(f"✓ Updated all {len(data)} prompts")

# Save updated dataset
print(f"Saving to: {output_file}")
with open(output_file, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("✓ Done!")
print(f"\nTo use this dataset, set: DATASET=\"sporc_235b_aligned%100\"")
