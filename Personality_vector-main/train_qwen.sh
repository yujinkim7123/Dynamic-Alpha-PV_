#!/usr/bin/env bash
# ============================================================
#  Qwen2.5-7B-Instruct 용 학습 스크립트
#  논문 Table 5의 하이퍼파라미터를 적용했어요!
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# 학습할 특성(trait)과 레벨(level)을 여기서 설정하세요!
TRAIT="extraversion"   # openness | conscientiousness | extraversion | agreeableness | neuroticism
LEVEL="high"           # high | low

personality-train \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --data-path wenkai-li/big5_chat \
  --output_dir "${ROOT_DIR}/outputs/qwen_${LEVEL}_${TRAIT}" \
  --batch_size 1 \
  --micro_batch_size 32 \
  --num_epochs 3 \
  --learning_rate 1e-5 \
  --cutoff_len 2048 \
  --train_on_inputs False \
  --add_eos_token False \
  --group_by_length False \
  --prompt_template_name qwen \
  --lr_scheduler cosine \
  --warmup_steps 40 \
  --wandb_run_name "qwen_${LEVEL}_${TRAIT}" \
  --trait "${TRAIT}" \
  --level "${LEVEL}"

# ============================================================
#  📌 사용법:
#    bash train_qwen.sh
#
#  📌 다른 특성을 학습하려면 위의 TRAIT, LEVEL 변수를 바꾸세요.
#    예시) TRAIT="openness" LEVEL="low" bash train_qwen.sh
# ============================================================
