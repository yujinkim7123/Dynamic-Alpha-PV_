
import os
import sys
import json
import logging
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────
OPENAI_API_KEY = ""    # ← 실제 키로 교체
RESULT_DIR     = ""
MERGE_METHOD   = "task_arithmetic"        # 단독 머징 → task_arithmetic
ALPHA_VALUE    = 1.0
N_REPEAT       = 5

# ⚠ phi_dir 경로는 실제 환경에 맞게 확인 후 수정!
MODELS = {
    "llama": {
        "base_model_path": "",
        "phi_dir":         "",
    },
    "qwen": {
        "base_model_path": "",
        "phi_dir":         "",
    },
}

BIG_FIVE    = ["OPN", "CON", "EXT", "AGR", "NEU"]
BFI_KEY_MAP = {
    "OPN": "openness",
    "CON": "conscientiousness",
    "EXT": "extraversion",
    "AGR": "agreeableness",
    "NEU": "neuroticism",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# 모듈 import
# ──────────────────────────────────────────────────────────────
from merger import DynamicMerger
from evaluator import bfi_eval
from openai import OpenAI


# ──────────────────────────────────────────────────────────────
# 핵심 함수들
# ──────────────────────────────────────────────────────────────
def get_base_bfi(model, tokenizer, openai_client) -> dict:
    """베이스 모델 BFI 측정 (머징 전 기준점)"""
    logger.info("[Base] 베이스 BFI 측정 중...")
    bfi_scores, _, _, _ = bfi_eval(
        model=model,
        tokenizer=tokenizer,
        openai_client=openai_client,
    )
    logger.info(f"[Base] BFI = {bfi_scores}")
    return bfi_scores


def measure_single_phi(dim, merger_obj, base_bfi):
    alpha = {d: 0.0 for d in BIG_FIVE}
    alpha[dim] = 1.0
    delta_list = {k: [] for k in BFI_KEY_MAP.values()}

    for rep in range(1, N_REPEAT + 1):
        print(f"  [{dim}] {rep}/{N_REPEAT}")

        # ── 머징 전 캐시 비우기 ──
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  VRAM 여유: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB")

        merged = merger_obj.merge(alpha)
        bfi, _, _, _ = run_eval(
            model=merged,
            tokenizer=merger_obj.tokenizer,
            bfi_meta=bfi_meta,
            openai_client=openai_client,
        )
        for k in BFI_KEY_MAP.values():
            delta_list[k].append(bfi.get(k, 0) - base_bfi.get(k, 0))
        merger_obj._reset_to_base()

        # ── 측정 후 캐시 비우기 ──
        gc.collect()
        torch.cuda.empty_cache()

        print(f"  [{dim}] BFI={bfi}")

    avg = {k: round(float(np.mean(v)), 4) for k, v in delta_list.items()}
    print(f"[{dim}] 평균 변화량={avg}")
    return avg

def build_C_matrix(results: dict) -> np.ndarray:
    """측정 결과 dict → 5×5 numpy C 행렬"""
    C = np.zeros((5, 5))
    for i, phi_dim in enumerate(BIG_FIVE):
        for j, bfi_dim in enumerate(BIG_FIVE):
            C[i][j] = results[phi_dim].get(BFI_KEY_MAP[bfi_dim], 0.0)
    return C


def compute_alpha_star(C: np.ndarray, delta_target: np.ndarray) -> np.ndarray:
    """CAAS 핵심: α* = pinv(C) × Δ_target"""
    return np.linalg.pinv(C) @ delta_target


def save_results(C: np.ndarray, results: dict, model_name: str):
    """C 행렬 저장 (npy + json)"""
    out_dir = os.path.join(RESULT_DIR, model_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(out_dir, "C_matrix.npy"), C)

    out = {
        "model":       model_name,
        "method":      MERGE_METHOD,
        "n_repeat":    N_REPEAT,
        "alpha_value": ALPHA_VALUE,
        "C_matrix": {
            "dims":   BIG_FIVE,
            "values": C.tolist(),
            "note":   "rows=phi merged, cols=BFI changed. diagonal=intended effect, off-diagonal=co-movement",
        },
        "raw_results": results,
    }
    with open(os.path.join(out_dir, "C_matrix.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ [{model_name}] 저장 완료 → {out_dir}")


def print_C_matrix(C: np.ndarray, model_name: str):
    """C 행렬 보기 좋게 출력"""
    print(f"\n{'='*60}")
    print(f"Co-movement Matrix C  [{model_name}]")
    print(f"{'='*60}")
    print(f"{'':8s}", end="")
    for dim in BIG_FIVE:
        print(f"{dim:>8s}", end="")
    print()
    for i, phi_dim in enumerate(BIG_FIVE):
        print(f"φ_{phi_dim:4s}", end="  ")
        for j in range(5):
            marker = "★" if i == j else " "
            print(f"{C[i][j]:>7.3f}{marker}", end="")
        print()
    print(f"{'='*60}")
    print("★ = 의도한 차원 / 나머지 = co-movement (간섭)")

    print(f"\n⚠ [{model_name}] 간섭 심한 쌍 Top 3:")
    pairs = [
        (abs(C[i][j]), BIG_FIVE[i], BIG_FIVE[j], C[i][j])
        for i in range(5) for j in range(5) if i != j
    ]
    for _, phi, bfi, val in sorted(pairs, reverse=True)[:3]:
        print(f"   φ_{phi} → BFI_{bfi}: {val:+.3f}")


def demo_alpha_star(C: np.ndarray, model_name: str):
    """α* 역산 데모 출력"""
    print(f"\n[CAAS 데모 — {model_name}] CON만 +0.5 올리고 싶을 때:")
    delta_target = np.array([0, 0.5, 0, 0, 0])
    alpha_star   = compute_alpha_star(C, delta_target)
    predicted    = C @ alpha_star

    print(f"  기존 α:   CON=0.5, 나머지=0.0")
    print(f"  CAAS α*: {dict(zip(BIG_FIVE, [round(a, 4) for a in alpha_star]))}")
    print(f"  예측 BFI 변화:")
    for dim, val in zip(BIG_FIVE, predicted):
        marker = " ← 목표" if dim == "CON" else ""
        print(f"    {dim}: {val:+.3f}{marker}")


def run_one_model(model_name: str, model_cfg: dict, openai_client) -> np.ndarray:
    """모델 하나 전체 파이프라인"""
    logger.info(f"\n{'#'*60}")
    logger.info(f"# 모델: {model_name.upper()}")
    logger.info(f"# base: {model_cfg['base_model_path']}")
    logger.info(f"# phi:  {model_cfg['phi_dir']}")
    logger.info(f"{'#'*60}")

    merger = DynamicMerger(
        base_model_path=model_cfg["base_model_path"],
        phi_dir=model_cfg["phi_dir"],
        method=MERGE_METHOD,
        dare_drop_rate=0.5,
        dare_rescale=True,
        dare_strategy="random",
        ties_trim_rate=0.7,
        scaling_coefficient=1.0,
    )

    base_bfi = get_base_bfi(merger.base_model, merger.tokenizer, openai_client)

    results = {}
    for dim in BIG_FIVE:
        results[dim] = measure_single_phi(
            dim=dim,
            merger=merger,
            tokenizer=merger.tokenizer,
            openai_client=openai_client,
            base_bfi=base_bfi,
        )

    C = build_C_matrix(results)
    print_C_matrix(C, model_name)
    demo_alpha_star(C, model_name)
    save_results(C, results, model_name)
    return C


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("CAAS Phase 1 — C 행렬 측정")
    logger.info(f"  모델:     {list(MODELS.keys())}")
    logger.info(f"  방법:     {MERGE_METHOD}")
    logger.info(f"  반복:     {N_REPEAT}회")
    logger.info(f"  예상시간: ~{len(MODELS) * len(BIG_FIVE) * N_REPEAT * 5}분")
    logger.info("=" * 60)

    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    all_C = {}
    for model_name, model_cfg in MODELS.items():
        all_C[model_name] = run_one_model(model_name, model_cfg, openai_client)

    # Llama vs Qwen 비교
    if "llama" in all_C and "qwen" in all_C:
        print(f"\n{'='*60}")
        print("Llama vs Qwen — Co-movement 비교 (차이 > 0.05인 것만)")
        print(f"{'='*60}")
        diff = all_C["llama"] - all_C["qwen"]
        found = False
        for i, phi in enumerate(BIG_FIVE):
            for j, bfi in enumerate(BIG_FIVE):
                if i != j and abs(diff[i][j]) > 0.05:
                    print(f"  φ_{phi}→BFI_{bfi}: Llama{all_C['llama'][i][j]:+.3f}  Qwen{all_C['qwen'][i][j]:+.3f}  diff{diff[i][j]:+.3f}")
                    found = True
        if not found:
            print("  두 모델 간 co-movement 패턴이 유사함")

    logger.info("\n✅ 전체 완료! 결과 → caas/llama/ & caas/qwen/")


if __name__ == "__main__":
    main()