#!/usr/bin/env python3
"""
run_one_method.py
=====================================================================
직교화 방법 "하나"만 골라서 C-matrix 측정을 수행합니다.
측정이 끝나면 (--keep-phi 옵션이 없는 한) 해당 phi 벡터 폴더를 삭제해서
디스크 공간을 확보합니다. 결과 JSON(std/CI/raw_runs 포함)은 항상 보존됩니다.

사용법:
    python run_one_method.py --method orig
    python run_one_method.py --method lowdin
    python run_one_method.py --method awd --keep-phi     # AWD는 이미 있으니 보존
    python run_one_method.py --method gs_opn_first
    python run_one_method.py --method gs_con_first
    python run_one_method.py --method mgs_opn_first
    python run_one_method.py --method mgs_con_first

환경변수 (실행 전 export 하거나 os.environ으로 설정):
    OPENAI_API_KEY   (필수)
    HF_TOKEN         (필요시)
"""

import argparse
import os
import shutil
import sys

# ── ★ 서버 실제 경로로 수정하세요 (기본값 = Llama) ★ ─────────────────────
PROJECT_ROOT    = "/workspace/code/Dynamic-Alpha-PV_-main"
BASE_MODEL_PATH_DEFAULT = "meta-llama/Llama-3.1-8B-Instruct"
PHI_DIR_ORIG    = f"{PROJECT_ROOT}/phi_vectors"
ORTHO_ROOT      = f"{PROJECT_ROOT}/ortho_phi_vectors"
RESULT_DIR      = f"{PROJECT_ROOT}/C_matrix_results"
# ──────────────────────────────────────────────────────────────────────

# 방법 이름 -> (모델명(결과 json 파일명), phi 폴더 경로)
METHOD_MAP = {
    "orig":         ("llama_task",           PHI_DIR_ORIG),
    "lowdin":       ("llama_ortho_lowdin",   f"{ORTHO_ROOT}/lowdin"),
    "awd":          ("llama_ortho_awd",      f"{ORTHO_ROOT}/awd"),
    "gs_opn_first": ("llama_gs_opn_first",   f"{ORTHO_ROOT}/gs_opn_first"),
    "gs_con_first": ("llama_gs_con_first",   f"{ORTHO_ROOT}/gs_con_first"),
    "mgs_opn_first":("llama_mgs_opn_first",  f"{ORTHO_ROOT}/mgs_opn_first"),
    "mgs_con_first":("llama_mgs_con_first",  f"{ORTHO_ROOT}/mgs_con_first"),
    # Qwen (High only)
    "qwen_orig":         ("qwen_task",             f"{PROJECT_ROOT}/qwen_phi_vectors"),
    "qwen_lowdin":       ("qwen_ortho_lowdin",     f"{PROJECT_ROOT}/qwen_ortho_phi_vectors/lowdin"),
    "qwen_awd":          ("qwen_ortho_awd",        f"{PROJECT_ROOT}/qwen_ortho_phi_vectors/awd"),
    "qwen_gs_opn_first": ("qwen_gs_opn_first",     f"{PROJECT_ROOT}/qwen_ortho_phi_vectors/gs_opn_first"),
    "qwen_gs_con_first": ("qwen_gs_con_first",     f"{PROJECT_ROOT}/qwen_ortho_phi_vectors/gs_con_first"),
    "qwen_mgs_opn_first":("qwen_mgs_opn_first",    f"{PROJECT_ROOT}/qwen_ortho_phi_vectors/mgs_opn_first"),
    "qwen_mgs_con_first":("qwen_mgs_con_first",    f"{PROJECT_ROOT}/qwen_ortho_phi_vectors/mgs_con_first"),
}


def main():
    parser = argparse.ArgumentParser(description="C-matrix 측정 (방법 1개)")
    parser.add_argument("--method", required=True, choices=list(METHOD_MAP.keys()))
    parser.add_argument("--keep-phi", action="store_true",
                         help="측정 후 phi 벡터 폴더를 삭제하지 않고 보존")
    parser.add_argument("--phi-dir", default=None,
                         help="phi 폴더 경로를 직접 지정 (기본값 대신 사용)")
    parser.add_argument("--base-model", default=None,
                         help="base 모델 경로 (기본값: Llama-3.1-8B-Instruct, qwen_* method는 자동으로 Qwen 사용)")
    args = parser.parse_args()

    model_name, phi_dir = METHOD_MAP[args.method]
    if args.phi_dir:
        phi_dir = args.phi_dir

    # base 모델 결정: --base-model 인자 > method가 qwen_로 시작하면 자동 Qwen > 기본값(Llama)
    if args.base_model:
        base_model_path = args.base_model
    elif args.method.startswith("qwen_"):
        base_model_path = "Qwen/Qwen2.5-7B-Instruct"
    else:
        base_model_path = BASE_MODEL_PATH_DEFAULT

    if not os.path.isdir(phi_dir):
        print(f"❌ phi 폴더가 없습니다: {phi_dir}")
        sys.exit(1)

    # 이미 측정된 결과가 있으면 건너뛰기 (안전한 재실행)
    result_path = os.path.join(RESULT_DIR, f"C_matrix_{model_name}.json")
    if os.path.exists(result_path):
        print(f"✅ 이미 측정 완료된 결과가 있습니다: {result_path}")
        print("   재측정하려면 해당 JSON 파일을 먼저 삭제하세요.")
        sys.exit(0)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")

    sys.path.insert(0, PROJECT_ROOT)
    import c_matrix_common as cm

    bfi_meta, openai_client = cm.get_clients()

    print(f"\n{'='*60}")
    print(f" 방법: {args.method}  ({model_name})")
    print(f" phi_dir: {phi_dir}")
    print(f" 결과 저장 위치: {result_path}")
    print(f"{'='*60}\n")

    merger_obj = cm.DynamicMerger(
        base_model_path=base_model_path,
        phi_dir=phi_dir,
        method="task_arithmetic",
        dare_drop_rate=0.5, dare_rescale=True,
        dare_strategy="random", ties_trim_rate=0.7,
        scaling_coefficient=1.0,
    )
    print(f"✅ merger 로드 완료 — phi_cache: {len(merger_obj.phi_cache)}개")

    model_type = "qwen" if args.method.startswith("qwen_") else "llama"
    cm.run_full_measurement(model_name, merger_obj, bfi_meta, openai_client, RESULT_DIR, model_type=model_type)

    # ── 메모리 정리 ──
    del merger_obj
    cm.clear_vram()

    # ── phi 벡터 삭제 (디스크 확보) ──
    if not args.keep_phi and args.method != "orig":
        # 원본(orig)은 앞으로도 계속 필요할 가능성이 높으므로 기본적으로 보존.
        # 직교화된 벡터들만 측정 후 삭제 대상으로 취급.
        print(f"\n🗑  phi 벡터 삭제 중: {phi_dir}")
        shutil.rmtree(phi_dir)
        print("✅ 삭제 완료. 디스크 공간 확보됨.")
    else:
        print(f"\n💾 phi 벡터 보존됨 (--keep-phi 또는 원본): {phi_dir}")

    print(f"\n🎉 [{args.method}] 전체 완료!")


if __name__ == "__main__":
    main()