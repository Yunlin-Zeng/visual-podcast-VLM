# Visual Podcast VLM

**SPoRC-VIST: A Benchmark for Evaluating Generative Natural Narrative in Vision-Language Models**

This repository contains the code to reproduce the data generation pipeline and model training for visual podcast generation using Qwen3-VL.

[[Paper (arXiv)]](https://arxiv.org/abs/XXXX.XXXXX) | [[WACV 2026 Workshop]](https://wacv2026-image-quality-workshop.github.io/)

## Overview

We present a novel pipeline for **end-to-end visual podcast generation** that transforms image sequences into natural, multi-speaker podcast dialogues. Our approach fine-tunes Qwen3-VL-32B on a curated dataset of 4,000 image-dialogue pairs using a **synthetic-to-real training strategy**:

- **Training**: High-quality podcast dialogues from the Structured Podcast Research Corpus (SPoRC) paired with synthetically generated images (Stable Diffusion 3.5)
- **Evaluation**: Real-world photo sequences from the Visual Storytelling Dataset (VIST)

### Key Results

| Metric | 32B Fine-tuned | 235B Base | Change |
|--------|----------------|-----------|--------|
| CLIPScore | 20.39 | 20.39 | 0.00 |
| Avg. Turn Length | **57.5** | 38.0 | +51% |
| Switch Rate (/1k words) | **16.0** | 27.0 | -41% |
| AI-Judge Win Rate | **>80%** | - | - |

Our fine-tuned 32B model achieves >80% win rate against the 235B base model in AI-as-a-Judge evaluations (Gemini 3 Pro, Claude Opus 4.5, GPT 5.2) while maintaining identical visual grounding (CLIPScore).

## Repository Structure

```
visual-podcast-VLM/
├── scripts_bedrock/          # Dataset generation pipeline (AWS Bedrock)
│   ├── generate_dataset.py   # Main pipeline: extract excerpts + generate images
│   ├── extract_excerpts.py   # Extract visualizable excerpts from SPoRC
│   └── generate_images.py    # Generate images with Stable Diffusion 3.5
├── scripts_yunlin/           # Training and evaluation scripts
│   ├── *_finetune_32b_*.sh   # Fine-tuning scripts for 32B model
│   ├── *_run_evaluation_*.sh # Evaluation scripts
│   ├── inference.py          # Inference script
│   └── *_convert_*.py        # Data format conversion utilities
├── scripts_235B/             # 235B model experiment scripts
├── qwen-vl-finetune/         # Fine-tuning framework (from Qwen3-VL)
├── data/                     # Training data in Qwen format (JSON)
├── evaluation_50_samples/    # Evaluation dataset (50 VIST sequences)
└── results/                  # Generated outputs and evaluation results
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Yunlin-Zeng/visual-podcast-VLM.git
cd visual-podcast-VLM

# Install dependencies
pip install torch==2.6.0 torchvision==0.21.0
pip install git+https://github.com/huggingface/transformers
pip install accelerate==1.7.0 deepspeed==0.17.1
pip install qwen-vl-utils==0.0.14
pip install flash_attn==2.7.4.post1
```

## Dataset Generation

The dataset generation pipeline uses AWS Bedrock for both text extraction (Claude Sonnet 4.5) and image generation (Stable Diffusion 3.5).

### Prerequisites

1. AWS credentials with Bedrock access
2. Access to SPoRC podcast transcripts

### Pipeline Overview

1. **Excerpt Extraction**: Claude Sonnet 4.5 identifies 600-800 word excerpts with rich visual descriptions from podcast transcripts
2. **Scene Generation**: For each excerpt, generate 5 detailed image prompts
3. **Image Synthesis**: Stable Diffusion 3.5 renders the prompts into images

```bash
# Set AWS credentials
export AWS_BEARER_TOKEN_BEDROCK="your_token_here"

# Run the dataset generation pipeline
cd scripts_bedrock
python generate_dataset.py
```

## Training

### Fine-tuning Qwen3-VL-32B with LoRA

```bash
cd scripts_yunlin

# Fine-tune on 4,004 samples (requires 8x A100 80GB GPUs)
bash 2025-11-28_finetune_32b_sporc_v3.sh
```

**Key Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Learning Rate | 4e-6 |
| Effective Batch Size | 32 |
| Epochs | 1 |
| NEFTune Noise Alpha | 5.0 |
| Weight Decay | 0.1 |
| Max Gradient Norm | 0.3 |

Training completes in ~12 hours on 8x A100 GPUs.

## Evaluation

### Run Inference on VIST Samples

```bash
cd scripts_yunlin

# Run evaluation on 50 VIST samples (parallel across 8 GPUs)
bash 2025-12-04_run_evaluation_32b_v3_new30.sh
```

### Inference Prompt

```
Generate a natural conversational podcast dialogue. Use the format Speaker 1:,
Speaker 2:, Speaker 3:, etc. for multiple speakers. Do not reference the images
or use phrases like 'our first image'. Write casual, authentic spoken dialogue
without introductions or sign-offs. The word count should be around 800 words.
```

## Pre-trained Models

| Model | Description | Download |
|-------|-------------|----------|
| Qwen3-VL-32B (Base) | Base model | [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |
| Qwen3-VL-32B (Fine-tuned) | LoRA adapter for podcast generation | Coming soon |

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zeng2026sporc,
  title={SPoRC-VIST: A Benchmark for Evaluating Generative Natural Narrative in Vision-Language Models},
  author={Zeng, Yunlin},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
  year={2026}
}
```

## Acknowledgments

This work builds upon:
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) - Base vision-language model
- [SPoRC](https://github.com/spotify-research/sporc) - Structured Podcast Research Corpus
- [VIST](https://visionandlanguage.net/VIST/) - Visual Storytelling Dataset

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
