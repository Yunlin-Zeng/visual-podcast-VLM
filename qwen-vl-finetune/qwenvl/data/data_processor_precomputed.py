"""
Modified data processor for loading precomputed vision embeddings.

This version loads precomputed vision encoder + MLP outputs from disk
instead of processing images on-the-fly, enabling training without
loading vision components.
"""

import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset

from .data_processor import (
    IGNORE_INDEX,
    rank0_print,
    update_processor_pixels,
)

logger = logging.getLogger(__name__)


class PrecomputedEmbeddingsDataset(Dataset):
    """
    Dataset that loads precomputed vision embeddings instead of processing images.

    This allows training the LLM without loading vision encoder + MLP,
    saving ~20-35GB per GPU in memory.
    """

    def __init__(self, processor, data_args, precomputed_dir: str):
        super().__init__()

        self.precomputed_dir = Path(precomputed_dir)
        if not self.precomputed_dir.exists():
            raise ValueError(f"Precomputed embeddings directory not found: {precomputed_dir}")

        # Load metadata
        metadata_file = self.precomputed_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        rank0_print(f"Loading precomputed embeddings from {precomputed_dir}")
        rank0_print(f"Metadata: {self.metadata}")

        # Load sample files (saved as sample_0000.npz, sample_0001.npz, etc.)
        self.sample_files = sorted(self.precomputed_dir.glob("sample_*.npz"))
        rank0_print(f"Found {len(self.sample_files)} sample files")

        if len(self.sample_files) == 0:
            raise ValueError(f"No sample_*.npz files found in {precomputed_dir}")

        # Build sample index: maps idx -> sample_file
        self.samples = self.sample_files

        rank0_print(f"Indexed {len(self.samples)} samples")

        # Load original annotations to get text conversations
        original_data_json = self.metadata['data_json']
        with open(original_data_json, 'r') as f:
            self.annotations = json.load(f)

        rank0_print(f"Loaded {len(self.annotations)} annotations")

        # Setup processor for text tokenization only
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Load one sample with precomputed vision embeddings.

        Returns:
            Dictionary with:
                - input_ids: tokenized text with placeholder for vision tokens
                - labels: labels for language modeling
                - vision_embeddings: precomputed vision features
                - vision_grid_thws: spatial grid info for RoPE
        """
        sample_file = self.samples[idx]

        # Load precomputed embeddings for this sample
        sample_data = np.load(sample_file, allow_pickle=True)

        sample_id = int(sample_data['sample_id'])
        vision_embeddings = sample_data['vision_embeddings']
        vision_grid_thws = sample_data['vision_grid_thws']

        # Convert to tensors
        vision_embeddings = torch.from_numpy(vision_embeddings).to(torch.bfloat16)
        vision_grid_thws = torch.from_numpy(vision_grid_thws)

        # Get text conversations from annotations
        annotation = self.annotations[sample_id]
        conversations = annotation['conversations']

        # Build messages for tokenization
        # We need to preserve <image> tokens in the text for proper positioning
        messages = []
        for conv in conversations:
            role = "user" if conv["from"] == "human" else "assistant"
            content = conv["value"]
            messages.append({"role": role, "content": content})

        # Tokenize text (without actually processing images)
        # The processor will create placeholders for <image> tokens
        text_result = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            # Don't pass images - we'll insert precomputed embeddings later
        )

        input_ids = text_result["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)

        # Create labels (same logic as original data_processor.py)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        input_ids_flat = input_ids[0].tolist()
        L = len(input_ids_flat)
        pos = 0
        while pos < L:
            if input_ids_flat[pos] == 77091:  # Assistant response start token
                ans_start = pos + 2
                ans_end = ans_start
                while ans_end < L and input_ids_flat[ans_end] != 151645:  # End token
                    ans_end += 1
                if ans_end < L:
                    labels[0, ans_start : ans_end + 2] = input_ids[0, ans_start : ans_end + 2]
                    pos = ans_end
            pos += 1

        return {
            "input_ids": input_ids.squeeze(0),
            "labels": labels.squeeze(0),
            "vision_embeddings": vision_embeddings,  # [num_vision_tokens, hidden_dim]
            "vision_grid_thws": vision_grid_thws,     # [num_images, 3]
        }


class PrecomputedDataCollator:
    """
    Data collator for batches with precomputed vision embeddings.

    This handles padding and batching of both text and precomputed vision features.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of instances with precomputed embeddings.
        """
        # Extract components
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        vision_embeddings = [inst["vision_embeddings"] for inst in instances]
        vision_grid_thws = [inst["vision_grid_thws"] for inst in instances]

        # Pad input_ids and labels
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        # Create attention mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Stack vision features (they should already be same shape per batch)
        # If they're different shapes, we'll need to handle that differently
        try:
            vision_embeddings = torch.stack(vision_embeddings, dim=0)
        except RuntimeError:
            # Different number of vision tokens per sample - keep as list
            pass

        try:
            vision_grid_thws = torch.stack(vision_grid_thws, dim=0)
        except RuntimeError:
            # Different number of images per sample - keep as list
            pass

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "vision_embeddings": vision_embeddings,
            "vision_grid_thws": vision_grid_thws,
        }


def create_precomputed_dataset(processor, data_args, precomputed_dir: str):
    """
    Factory function to create dataset with precomputed embeddings.

    Args:
        processor: HuggingFace processor (used for tokenization only)
        data_args: Data arguments
        precomputed_dir: Directory containing precomputed embeddings

    Returns:
        PrecomputedEmbeddingsDataset instance
    """
    return PrecomputedEmbeddingsDataset(processor, data_args, precomputed_dir)
