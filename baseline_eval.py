import os
import json
import torch
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

# 기존 업로드하신 evaluator.py의 핵심 함수와 메타데이터 로드
try:
    from evaluator import run_eval, _BFI_META
except ImportError:
    print("❌ evaluator.py를 찾을 수 없습니다. 파일이 같은 경로에 있는지 확인해주세요.")

# =============================================================================
# [설정] Llama 모델 및 API 키 설정
# =============================================================================
# 예: "meta-llama/Llama-3.1-8B-Instruct" 또는 로컬 절대 경로
MODEL_PATH = "" 

# OpenAI API 키 직접 입력
MY_OPENAI_KEY = "" 

# 반복 횟수
NUM_RUNS = 5

# 결과 저장 경로
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"llama_baseline_bfi_{TIMESTAMP}.json"

# =============================================================================
# [검토] Llama 모델 로딩 및 환경 최적화
# =============================================================================
def load_llama_and_tokenizer(path):
    print(f"📦 Llama 모델 로드 중: {path}")
    
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    
    # 꼼꼼한 체크 1: Llama는 pad_token이 없는 경우가 많아 eos_token으로 대체 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("💡 Pad token이 없어 EOS token으로 설정했습니다.")

    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.bfloat16, # Llama-3 계열은 bfloat16 권장
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

# =============================================================================
# [실행] 벤치마크 루프
# =============================================================================
def run_benchmark():
    model, tokenizer = load_llama_and_tokenizer(MODEL_PATH)
    client = OpenAI(api_key=MY_OPENAI_KEY)

    all_runs_data = []
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    score_accumulator = {t: [] for t in traits}

    print(f"\n📊 BFI 벤치마크 시작 (Llama Baseline / {NUM_RUNS}회 반복)")

    for i in range(NUM_RUNS):
        run_idx = i + 1
        print(f"\n--- [{run_idx}/{NUM_RUNS}] 회차 진행 중 ---")
        
        # evaluator.py의 run_eval 호출
        # bfi_scores(결과 점수), qa_pairs(답변 원문), assessment_raw(GPT 채점 근거)
        bfi_scores, liwc_scores, qa_pairs, assessment_raw = run_eval(
            model=model, 
            tokenizer=tokenizer,
            bfi_meta=_BFI_META, 
            openai_client=client
        )
        
        # 데이터 기록
        run_result = {
            "run_number": run_idx,
            "bfi_scores": bfi_scores,
            "liwc_scores": liwc_scores,
            "responses": qa_pairs,           # 모델의 실제 답변 전수 기록
            "gpt_assessment": assessment_raw  # GPT의 채점 근거 기록
        }
        all_runs_data.append(run_result)
        
        # 평균 계산용 점수 저장
        for trait in traits:
            score_accumulator[trait].append(bfi_scores.get(trait, 0))
            
        print(f"✅ {run_idx}회차 완료 (Scores: {bfi_scores})")

    # 최종 평균 계산
    final_averages = {t: round(float(np.mean(score_accumulator[t])), 4) for t in traits}

    # 전체 리포트 생성
    report = {
        "metadata": {
            "model_path": MODEL_PATH,
            "total_runs": NUM_RUNS,
            "timestamp": TIMESTAMP,
            "config": "Pure Baseline (No Merging)"
        },
        "final_average_scores": final_averages,
        "detailed_runs": all_runs_data
    }

    # JSON 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n" + "="*55)
    print(f"✨ 실험 완료! 결과가 성공적으로 저장되었습니다.")
    print(f"📁 파일: {OUTPUT_FILE}")
    print(f"🏆 최종 평균 BFI: {final_averages}")
    print("="*55)

if __name__ == "__main__":
    run_benchmark()