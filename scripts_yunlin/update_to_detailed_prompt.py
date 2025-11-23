"""
Update SPoRC training data to use FULL DETAILED prompt from Nov 9 Test 1
(includes examples, question patterns, good/bad visual descriptions)
"""
import json

# FULL DETAILED prompt from Nov 9 Test 1 (the one that worked well)
DETAILED_PROMPT = """You are creating a podcast transcript in a Q&A format between two hosts based on the image I uploaded. They narrate a descriptive and detailed story based on the images. They can ask questions. Based on the visual information from these 5 images, create an engaging conversational podcast transcript.

CRITICAL REQUIREMENTS:

1. STRUCTURE (REQUIRED):

* One host guides the talk. Sometimes either host can ask questions.
* Both hosts can provide rich visual descriptions
* Questions should follow the story arc: setting → characters (if any) → action → details → conclusion

   Question pattern example:

* Q1: "What's happening here? Where does this take place?"
* Q2: "Tell me about the people involved. What are they wearing?"
* Q3: "What happens next in the story?"
* Q4: "Are there any other interesting details?"
* Q5: "How does this story end?"

2. FORMAT (NON-NEGOTIABLE):

* Do NOT cut the conversation short
* Each answer should be substantial

3. VISUAL DETAILS IN ANSWERS (REQUIRED):

   Must include SPECIFIC, RICH visual descriptions:

   ✓ GOOD examples (be this specific):

* "black jacket with bright yellow racing stripes down the sleeves"
* "light blue plaid boxer shorts with white cross-hatching pattern"
* "lace curtains filtering soft morning light"
* "birthday card with 'HAPPY BIRTHDAY' written in bold red marker"
* "wooden table with warm honey-colored finish"

   ✗ BAD examples (avoid generic descriptions):

* "nice jacket" (too vague)
* "blue boxers" (missing pattern detail)
* "pretty kitchen" (no specific details)

   Details to include in answers:

* Exact colors and patterns
* Specific clothing details and materials
* Textures and finishes
* Environmental elements (decorations, furniture, wall items)
* Spatial relationships
* Lighting and atmosphere

4. NATURAL DIALOGUE (REQUIRED):

* DO NOT mention "images", "photos", "pictures"
* Tell the story naturally
* Questions should feel curious and engaged
* Answers should be descriptive and informative

   ✓ GOOD:
   What's happening in this birthday scene? Where does this take place?
   This unfolds in the coziest home kitchen. We've got delicate white lace curtains filtering natural light, walls covered with family photos and postcards, even a heart decoration made from blue handprints.

   ✗ BAD:
   What do you see in the first image?
   In the image, there is a kitchen with curtains.


5. (optional) STORY PROGRESSION THROUGH QUESTIONS:

* Each answer should build on previous information

Please generate a transcript in a conversational style, with a dramatic tone, in podcast format, with length between 600-700 words (do NOT exceed 700 words), and elaborate detail level.

Generate a two-host conversation about this visual sequence."""

# Load original training data (with simplified prompt)
print("Loading original dataset with simplified prompt...")
with open('data/qwen_training_data_sporc_excerpt_working_prompt.json', 'r') as f:
    original_data = json.load(f)

print(f"Total samples: {len(original_data)}")

# Update all samples with detailed prompt
updated_data = []
for i, sample in enumerate(original_data):
    num_images = len(sample["image"])
    image_tags = "<image>" * num_images

    updated_sample = {
        "image": sample["image"],
        "conversations": [
            {
                "from": "human",
                "value": f"{image_tags}\n{DETAILED_PROMPT}"
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
output_file = 'data/qwen_training_data_sporc_detailed_prompt.json'
with open(output_file, 'w') as f:
    json.dump(updated_data, f, indent=2)

print(f"\n✓ Updated dataset saved to: {output_file}")
print(f"  - Total samples: {len(updated_data)}")
print(f"  - All samples now use FULL DETAILED prompt from Nov 9 Test 1")
print(f"  - Includes: question patterns, good/bad examples, dramatic tone, 600-700 words")
print(f"\nKey differences from simplified prompt:")
print(f"  ✓ Question pattern examples")
print(f"  ✓ Good/bad visual description examples")
print(f"  ✓ 'dramatic tone' and 'elaborate detail level'")
print(f"  ✓ '600-700 words (do NOT exceed 700 words)' vs 'around 800 words'")
