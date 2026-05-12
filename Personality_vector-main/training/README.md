# personality_llm_pipeline


## 설치

```bash
cd personality_llm_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

필요 환경변수:

```bash
export HF_TOKEN=...
export WANDB_API_KEY=...
export OPENAI_API_KEY=...
```

## 학습

설치 후에는 `personality-train` 명령을 사용할 수 있습니다.

```bash
personality-train \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --data-path wenkai-li/big5_chat \
  --output_dir outputs/HIGH_EXT_pilot \
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
```

설치 없이 바로 실행하려면:

```bash
PYTHONPATH=src python -m personality_llm_pipeline.cli \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --data-path wenkai-li/big5_chat \
  --output_dir outputs/HIGH_EXT_pilot \
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
```

예시 스크립트:

```bash
bash scripts/train_example.sh
```

## 평가

```bash
python personality_tests.py \
  --llama_model_path outputs/HIGH_EXT_pilot \
  --questionnaire_name BFI \
  --character myagent \
  --agent_type mycustom \
  --agent_llm gpt-3.5-turbo \
  --evaluator_llm gpt-4o \
  --eval_method interview_batch \
  --system_prompt "Imagine you are a real person rather than a language model."
```
