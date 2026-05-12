#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

personality-train \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --data-path wenkai-li/big5_chat \
  --output_dir "${ROOT_DIR}/outputs/HIGH_EXT_pilot" \
  --batch_size 1 \
  --micro_batch_size 64 \
  --num_epochs 3 \
  --learning_rate 5e-6 \
  --cutoff_len 2028 \
  --train_on_inputs False \
  --add_eos_token False \
  --group_by_length False \
  --prompt_template_name llama \
  --lr_scheduler cosine \
  --warmup_steps 40 \
  --wandb_run_name EXT_high_0210 \
  --trait extraversion \
  --level high
