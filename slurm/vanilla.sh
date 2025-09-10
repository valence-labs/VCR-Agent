#!/bin/bash
#SBATCH --job-name=two_stage
#SBATCH --output=sbatch_log/%x.out
#SBATCH --error=sbatch_log/%x.err
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

date

nvidia-smi


uv run src/explain/llm/generate.py \
    --experiment_name vanilla
