#!/usr/bin/env bash
# ============================================================
#  LLaMA-3.1-8B-Instruct 용 학습 스크립트
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# 학습할 특성(trait)과 레벨(level)을 여기서 설정하세요!
TRAIT="neuroticism"
LEVEL="high"

personality-train \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --data-path wenkai-li/big5_chat \
  --output_dir "${ROOT_DIR}/outputs/llama_${LEVEL}_${TRAIT}" \
  --batch_size 1 \
  --micro_batch_size 64 \
  --num_epochs 3 \
  --learning_rate 5e-6 \
  --cutoff_len 2048 \
  --train_on_inputs False \
  --add_eos_token False \
  --group_by_length False \
  --prompt_template_name llama \
  --lr_scheduler cosine \
  --warmup_steps 40 \
  --wandb_run_name "llama_${LEVEL}_${TRAIT}" \
  --trait "${TRAIT}" \
  --level "${LEVEL}"

# ============================================================
#  📌 사용법: bash train_llama.sh
# ============================================================