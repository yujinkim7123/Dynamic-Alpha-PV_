"""
c_matrix_common.py
=====================================================================
C-matrix 측정용 공통 함수 모음.
run_one_method.py에서 import해서 사용합니다.
=====================================================================
"""

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI

from evaluator import _load_bfi_json, assess, collect_bfi_qa, BFI_KEYS
from merger import DynamicMerger

BIG_FIVE = ["OPN", "CON", "EXT", "AGR", "NEU"]
BFI_KEY_MAP = {
    "OPN": "openness",
    "CON": "conscientiousness",
    "EXT": "extraversion",
    "AGR": "agreeableness",
    "NEU": "neuroticism",
}

SYSTEM_PROMPT = (
    "Imagine you are a real person rather than a language model, "
    "and you're asked by the following question. "
    "Never mention AI, language models, or databases."
)
CHARACTER_NAME = "John"
EXPERIMENTER   = "<the experimenter>"
EVAL_METHOD    = "interview_batch"
LANGUAGE       = "en"
EVALUATOR_MODEL = "gpt-4o"   # 논문 Section 5.1 명시

N_REPEAT = 5   # 논문 Eq.14/15 확정값. 임의로 낮추지 말 것.


def get_clients():
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY가 설정되지 않았습니다."
    bfi_meta = _load_bfi_json()
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return bfi_meta, openai_client


def run_eval(model, tokenizer, bfi_meta, openai_client, model_type="llama"):
    """BFI-44 문항 인터뷰 수집 + GPT-4o 채점"""
    bfi_questions_dict = bfi_meta["questions"]
    questionnaire = []
    for idx, q in bfi_questions_dict.items():
        q_copy = dict(q)
        q_copy["id"] = idx
        questionnaire.append(q_copy)
    bfi_questions = list(bfi_questions_dict.values())

    qa_pairs = collect_bfi_qa(
        model=model,
        tokenizer=tokenizer,
        bfi_questions=bfi_questions,
        system_prompt=SYSTEM_PROMPT,
        model_type=model_type,
        language=LANGUAGE,
    )

    questionnaire_results = []
    for q_meta, qa in zip(questionnaire, qa_pairs):
        questionnaire_results.append({
            "id":            q_meta["id"],
            "question":      qa["question"],
            "response_open": qa["response_open"],
            "query_style":   "interview",
        })

    assessment_results = assess(
        character_aliases      = [CHARACTER_NAME],
        experimenter            = EXPERIMENTER,
        questionnaire_results   = questionnaire_results,
        questionnaire            = questionnaire,
        questionnaire_metadata  = bfi_meta,
        eval_method              = EVAL_METHOD,
        language                 = LANGUAGE,
        evaluator_llm            = EVALUATOR_MODEL,
        nth_test                 = 0,
        agent_llm                = model_type,
        openai_client             = openai_client,
        evaluator_model           = EVALUATOR_MODEL,
    )

    bfi_scores = {}
    for dim, res in assessment_results.items():
        if dim == "error_counts":
            continue
        key = BFI_KEYS.get(dim, dim.lower())
        bfi_scores[key] = round(res["score"], 3)

    return bfi_scores, qa_pairs, assessment_results, None


def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()


def get_base_bfi(merger_obj, bfi_meta, openai_client, model_type="llama"):
    bfi, _, _, _ = run_eval(
        model=merger_obj.base_model,
        tokenizer=merger_obj.tokenizer,
        bfi_meta=bfi_meta,
        openai_client=openai_client,
        model_type=model_type,
    )
    print(f"[Base BFI] {bfi}")
    return bfi


def measure_single_phi(dim, merger_obj, base_bfi, bfi_meta, openai_client, n_repeat=N_REPEAT, model_type="llama"):
    """논문 Eq.13-15 재현. std/CI/raw_runs까지 전부 반환."""
    alpha = {d: 0.0 for d in BIG_FIVE}
    alpha[dim] = 1.0

    delta_list = {k: [] for k in BFI_KEY_MAP.values()}

    for rep in range(1, n_repeat + 1):
        print(f"  [{dim}] run {rep}/{n_repeat}")
        clear_vram()
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"    VRAM 여유: {free_gb:.1f} GB")

        merged = merger_obj.merge(alpha)
        bfi, _, _, _ = run_eval(
            model=merged,
            tokenizer=merger_obj.tokenizer,
            bfi_meta=bfi_meta,
            openai_client=openai_client,
            model_type=model_type,
        )
        for k in BFI_KEY_MAP.values():
            delta_list[k].append(bfi.get(k, 0) - base_bfi.get(k, 0))

        clear_vram()
        print(f"    [{dim}] run {rep} BFI={bfi}")

    avg  = {k: round(float(np.mean(v)), 4) for k, v in delta_list.items()}
    std  = {k: round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0
            for k, v in delta_list.items()}
    ci95 = {k: round(float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v))), 4) if len(v) > 1 else 0.0
            for k, v in delta_list.items()}

    print(f"[{dim}] avg={avg}")
    print(f"[{dim}] std={std}")

    return {
        "avg": avg,
        "std": std,
        "ci95_half_width": ci95,
        "n_repeat": n_repeat,
        "raw_runs": {k: [round(float(x), 4) for x in v] for k, v in delta_list.items()},
    }


def build_C_matrix(results: dict) -> np.ndarray:
    C = np.zeros((5, 5))
    for i, phi_dim in enumerate(BIG_FIVE):
        for j, bfi_dim in enumerate(BIG_FIVE):
            C[i][j] = results[phi_dim]["avg"].get(BFI_KEY_MAP[bfi_dim], 0.0)
    return C


def print_C_matrix(C: np.ndarray, model_name: str):
    print(f"\n{'='*60}")
    print(f"C-matrix [{model_name}]")
    print(f"{'':8}", *[f"{d:>8}" for d in BIG_FIVE])
    for i, phi in enumerate(BIG_FIVE):
        row = f"phi_{phi:4} "
        for j in range(5):
            row += f"{C[i][j]:>7.3f}{'*' if i == j else ' '}"
        print(row)
    print(f"{'='*60}")


def save_results(results: dict, C: np.ndarray, model_name: str, base_bfi: dict, result_dir: str):
    out_dir = Path(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"C_matrix_{model_name}.json"

    out = {
        "model": model_name,
        "base_bfi": base_bfi,
        "n_repeat": N_REPEAT,
        "C": C.tolist(),
        "raw": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"✅ 저장 완료 -> {out_path}")
    return out_path


def run_full_measurement(model_name: str, merger_obj, bfi_meta, openai_client, result_dir: str, model_type: str = "llama"):
    print(f"\n{'#'*60}\n# {model_name.upper()} 측정 시작 ({time.strftime('%H:%M:%S')})\n{'#'*60}")

    base_bfi = get_base_bfi(merger_obj, bfi_meta, openai_client, model_type=model_type)

    results = {}
    for dim in BIG_FIVE:
        results[dim] = measure_single_phi(dim, merger_obj, base_bfi, bfi_meta, openai_client, model_type=model_type)

    C = build_C_matrix(results)
    print_C_matrix(C, model_name)
    save_results(results, C, model_name, base_bfi, result_dir)

    print(f"# {model_name.upper()} 완료 ({time.strftime('%H:%M:%S')})")
    return C, results