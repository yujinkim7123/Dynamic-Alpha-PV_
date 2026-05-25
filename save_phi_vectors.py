#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
save_phi_vectors.py

[ 역할 ]
Big Five 성격 차원별로 SFT된 10개 finetuned 모델에서
phi(φ) 벡터를 계산해서 .pt 파일로 저장한다.

    φ_i = θ_finetuned_i − θ_pretrained   ← TaskVector가 이 뺄셈을 해줌

[ 실행 방법 ]
python save_phi_vectors.py \
    --base_model_path  /path/to/Qwen2.5-7B-Instruct \
    --ft_model_dir     /path/to/Personality_vector-main_qwen_runpob/outputs \
    --output_dir       ./phi_vectors

[ 저장 결과 ]
phi_vectors/
    phi_OPN_High.pt       ← qwen_high_openness 에서 추출
    phi_OPN_Low.pt        ← qwen_low_openness 에서 추출
    phi_CON_High.pt       ← qwen_high_conscientiousness 에서 추출
    phi_CON_Low.pt        ← qwen_low_conscientiousness 에서 추출
    phi_EXT_High.pt       ← qwen_high_extraversion 에서 추출
    phi_EXT_Low.pt        ← qwen_low_extraversion 에서 추출
    phi_AGR_High.pt       ← qwen_high_agreeableness 에서 추출
    phi_AGR_Low.pt        ← qwen_low_agreeableness 에서 추출
    phi_NEU_High.pt       ← qwen_high_neuroticism 에서 추출
    phi_NEU_Low.pt        ← qwen_low_neuroticism 에서 추출
    phi_manifest.json     ← 메타정보 (언제 만들었는지, base 모델명 등)

[ 나중에 불러올 때 ]
    phi_dict = torch.load("phi_vectors/phi_CON_High.pt")
    phi = TaskVector(task_vector_param_dict=phi_dict)
"""

import argparse
import json
import logging
import os
import time

import torch
from transformers import AutoModelForCausalLM

# 기존 프로젝트 코드 그대로 사용
from task_vector import TaskVector
from utils.utils import set_random_seed


# ─────────────────────────────────────────────
# 저장할 10개 벡터 이름 정의
# (Big Five 5개 차원 × High/Low 2개 = 10개)
# ─────────────────────────────────────────────
PERSONALITY_KEYS = [
    "OPN_High",   # Openness High
    "OPN_Low",    # Openness Low
    "CON_High",   # Conscientiousness High
    "CON_Low",    # Conscientiousness Low
    "EXT_High",   # Extraversion High
    "EXT_Low",    # Extraversion Low
    "AGR_High",   # Agreeableness High
    "AGR_Low",    # Agreeableness Low
    "NEU_High",   # Neuroticism High
    "NEU_Low",    # Neuroticism Low
]

# finetuned 모델 폴더명 매핑
# --ft_model_dir (예: .../Personality_vector-main_qwen_runpob/outputs/) 아래
# 실제 폴더명과 우리 내부 키를 연결
FT_FOLDER_MAP = {
    "OPN_High": "llama_high_openness",
    "OPN_Low":  "llama_low_openness",
    "CON_High": "llama_high_conscientiousness",
    "CON_Low":  "llama_low_conscientiousness",
    "EXT_High": "llama_high_extraversion",
    "EXT_Low":  "llama_low_extraversion",
    "AGR_High": "llama_high_agreeableness",
    "AGR_Low":  "llama_low_agreeableness",
    "NEU_High": "llama_high_neuroticism",
    "NEU_Low":  "llama_low_neuroticism",
}


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_base_model(base_model_path: str, device: str = "cpu"):
    """
    Base(pretrained) 모델 로드
    - bfloat16으로 로드해서 메모리 절약
    - eval 모드 고정 (학습 안 함)
    """
    logger.info(f"Base 모델 로딩 중: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    logger.info("Base 모델 로딩 완료!")
    return model


def load_finetuned_model(ft_model_path: str, device: str = "cpu"):
    """
    Finetuned 모델 로드 (한 번 쓰고 바로 삭제함)
    """
    model = AutoModelForCausalLM.from_pretrained(
        ft_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model


def compute_and_save_phi(
    key: str,
    ft_model_path: str,
    base_model,
    output_dir: str,
    exclude_param_names_regex: list,
):
    """
    φ_i = θ_finetuned_i − θ_base 계산 후 .pt 파일로 저장

    [ 핵심 흐름 ]
    1. finetuned 모델 로드
    2. TaskVector로 뺄셈 (φ 계산)
    3. phi.task_vector_param_dict 를 .pt 파일로 저장
    4. finetuned 모델 메모리에서 삭제 (base는 보존!)
    """
    logger.info(f"[{key}] finetuned 모델 로딩: {ft_model_path}")
    ft_model = load_finetuned_model(ft_model_path)

    logger.info(f"[{key}] φ 계산 중 (θ_finetuned − θ_base)...")
    # ← 기존 task_vector.py 코드 그대로 사용!
    phi = TaskVector(
        pretrained_model=base_model,
        finetuned_model=ft_model,
        exclude_param_names_regex=exclude_param_names_regex,
    )

    # 저장 경로
    save_path = os.path.join(output_dir, f"phi_{key}.pt")

    # phi.task_vector_param_dict 구조:
    # {
    #   "model.layers.0.self_attn.q_proj.weight": tensor(...),
    #   "model.layers.0.self_attn.k_proj.weight": tensor(...),
    #   ...  (수억 개의 파라미터 차이값들)
    # }
    torch.save(phi.task_vector_param_dict, save_path)
    logger.info(f"[{key}] 저장 완료 → {save_path}")

    # finetuned 모델 즉시 삭제 (base는 절대 건드리지 않음!)
    del ft_model
    del phi
    torch.cuda.empty_cache()

    return save_path


def verify_phi(save_path: str, base_model, exclude_param_names_regex: list):
    """
    저장된 φ 파일이 정상인지 검증
    - 로드 후 파라미터 개수 / 첫 번째 파라미터 norm 출력
    """
    phi_dict = torch.load(save_path, map_location="cpu")
    phi = TaskVector(task_vector_param_dict=phi_dict)

    num_params = len(phi.task_vector_param_dict)
    first_key = next(iter(phi.task_vector_param_dict))
    first_norm = phi.task_vector_param_dict[first_key].float().norm().item()

    logger.info(f"  검증 OK — 파라미터 수: {num_params}, 첫 파라미터 norm: {first_norm:.6f}")

    del phi_dict, phi


def main():
    parser = argparse.ArgumentParser(description="φ 벡터 10개를 .pt 파일로 저장")

    parser.add_argument(
        "--base_model_path", type=str, required=True,
        help="Base(pretrained) 모델 경로"
    )
    parser.add_argument(
        "--ft_model_dir", type=str, required=True,
        help="Finetuned 모델들이 들어있는 상위 폴더 경로\n"
             "예: /models/finetuned/ 아래에 OPN_High/, OPN_Low/ 등 폴더가 있어야 함"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./phi_vectors",
        help="φ .pt 파일을 저장할 폴더 (없으면 자동 생성)"
    )
    parser.add_argument(
        "--exclude_params", type=str, nargs="*", default=[],
        help="φ 계산에서 제외할 파라미터 정규표현식\n"
             "예: --exclude_params 'lm_head' 'embed_tokens'"
    )
    parser.add_argument(
        "--keys", type=str, nargs="*", default=PERSONALITY_KEYS,
        help="저장할 키 목록 (기본값: 10개 전부)\n"
             "예: --keys CON_High CON_Low  ← 특정 벡터만 저장할 때"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true", default=True,
                        help="저장 후 파일 검증 여부")

    args = parser.parse_args()

    set_random_seed(args.seed)

    # 출력 폴더 생성
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"출력 폴더: {args.output_dir}")

    # ── 1. Base 모델 로드 (1번만!) ──────────────────────
    base_model = load_base_model(args.base_model_path)

    # ── 2. 10개 벡터 순서대로 처리 ─────────────────────
    manifest = {
        "base_model_path": args.base_model_path,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exclude_params": args.exclude_params,
        "phi_files": {}
    }

    for key in args.keys:
        # finetuned 모델 폴더 경로 조합
        # 예: /models/finetuned/OPN_High
        ft_path = os.path.join(args.ft_model_dir, FT_FOLDER_MAP[key])

        if not os.path.exists(ft_path):
            logger.warning(f"[{key}] 폴더 없음, 건너뜀: {ft_path}")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"처리 중: {key} ({args.keys.index(key)+1}/{len(args.keys)})")
        logger.info(f"{'='*50}")

        save_path = compute_and_save_phi(
            key=key,
            ft_model_path=ft_path,
            base_model=base_model,
            output_dir=args.output_dir,
            exclude_param_names_regex=args.exclude_params,
        )

        # 검증
        if args.verify:
            verify_phi(save_path, base_model, args.exclude_params)

        manifest["phi_files"][key] = save_path

    # ── 3. 메타정보 저장 ────────────────────────────────
    manifest_path = os.path.join(args.output_dir, "phi_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"\n메타정보 저장 완료 → {manifest_path}")

    # ── 4. 최종 요약 출력 ────────────────────────────────
    logger.info("\n" + "="*50)
    logger.info("저장된 φ 파일 목록:")
    for key, path in manifest["phi_files"].items():
        size_mb = os.path.getsize(path) / (1024**2)
        logger.info(f"  {key:12s} → {path}  ({size_mb:.1f} MB)")
    logger.info("="*50)
    logger.info("완료! 이제 save_phi_vectors.py는 다시 실행할 필요 없음.")
    logger.info("머징 시에는 phi_vectors/*.pt 파일만 로드해서 사용하면 됨.")


# 로거는 모듈 레벨에서 한 번만 생성
logger = setup_logger()

if __name__ == "__main__":
    main()
