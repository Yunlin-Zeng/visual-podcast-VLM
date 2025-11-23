#!/bin/bash
# Wrapper to run full training on GPUs 4-7
export CUDA_VISIBLE_DEVICES=4,5,6,7
bash scripts_yunlin/finetune_8b_sporc.sh
