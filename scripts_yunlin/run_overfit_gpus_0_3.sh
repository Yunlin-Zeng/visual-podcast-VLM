#!/bin/bash
# Wrapper to run overfit training on GPUs 0-3
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts_yunlin/overfit_8b_single_sample.sh
