#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_three_vector_simultaneous.py
=====================================================================
3-vector(NEU+OPN+AGR) 동시 적용 검증.
2-vector 검증(run_two_vector_lowdin_only.py)과 동일한 안전 패턴 사용:
  - 참고용 단독 측정 없음 (메모리/시간 절약)
  - 매 iteration마다 명시적 삭제 + gc.collect() + empty_cache()

[ 실행 방법 - 원본 ]
    python run_three_vector_simultaneous.py \
        --base_model_path meta-llama/Llama-3.1-8B-Instruct \
        --phi_dir /path/to/phi_vectors \
        --result_dir /path/to/C_matrix_results \
        --label original \
        --dims NEU OPN AGR

[ 실행 방법 - Löwdin ]
    python run_three_vector_simultaneous.py \
        --base_model_path meta-llama/Llama-3.1-8B-Instruct \
        --phi_dir /path/to/ortho_phi_vectors/lowdin \
        --result_dir /path/to/C_matrix_results \
        --label lowdin \
        --dims NEU OPN AGR
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--phi_dir", required=True)
    parser.add_argument("--result_dir", default="./C_matrix_results")
    parser.add_argument("--label", required=True, help="original 또는 lowdin 등")
    parser.add_argument("--dims", nargs="+", required=True, help="동시 적용할 차원들 (예: NEU OPN AGR)")
    parser.add_argument("--n_repeat", type=int, default=5)
    parser.add_argument("--project_root", default="/workspace/code/Dynamic-Alpha-PV_-main")
    args = parser.parse_args()

    sys.path.insert(0, args.project_root)
    import c_matrix_common as cm

    os.makedirs(args.result_dir, exist_ok=True)

    bfi_meta, openai_client = cm.get_clients()
    BIG_FIVE = cm.BIG_FIVE
    BFI_KEY_MAP = cm.BFI_KEY_MAP

    dims = args.dims
    assert all(d in BIG_FIVE for d in dims), f"dims는 {BIG_FIVE} 중이어야 합니다"
    dims_label = "_".join(dims)

    alpha_simultaneous = {d: 0.0 for d in BIG_FIVE}
    for d in dims:
        alpha_simultaneous[d] = 1.0

    print(f"{'='*60}")
    print(f" [{args.label}] {len(dims)}-vector 동시 적용: {' + '.join(dims)}")
    print(f" alpha = {alpha_simultaneous}")
    print(f"{'='*60}")

    gc.collect()
    torch.cuda.empty_cache()

    merger_obj = cm.DynamicMerger(
        base_model_path=args.base_model_path,
        phi_dir=args.phi_dir,
        method="task_arithmetic",
        dare_drop_rate=0.5, dare_rescale=True,
        dare_strategy="random", ties_trim_rate=0.7,
        scaling_coefficient=1.0,
    )
    print(f"merger 로드 완료 - phi_cache: {len(merger_obj.phi_cache)}개")

    base_bfi = cm.get_base_bfi(merger_obj, bfi_meta, openai_client)
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{' + '.join(dims)} 동시 적용, {args.n_repeat}회 반복 측정...")
    delta_list = {k: [] for k in BFI_KEY_MAP.values()}

    for rep in range(1, args.n_repeat + 1):
        print(f"  [동시적용] run {rep}/{args.n_repeat}")
        gc.collect()
        torch.cuda.empty_cache()

        merged = merger_obj.merge(alpha_simultaneous)
        bfi, _, _, _ = cm.run_eval(
            model=merged, tokenizer=merger_obj.tokenizer,
            bfi_meta=bfi_meta, openai_client=openai_client,
        )
        for k in BFI_KEY_MAP.values():
            delta_list[k].append(bfi.get(k, 0) - base_bfi.get(k, 0))

        del merged
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [동시적용] run {rep} BFI={bfi}")

    avg = {k: round(float(np.mean(v)), 4) for k, v in delta_list.items()}
    std = {k: round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0
           for k, v in delta_list.items()}
    ci95 = {k: round(float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v))), 4) if len(v) > 1 else 0.0
            for k, v in delta_list.items()}

    print(f"\n[{args.label}] 동시 적용 평균 변화량: {avg}")
    print(f"[{args.label}] std: {std}")

    # 간섭: 적용한 dims가 아닌 나머지 차원들의 평균 절대 변화량
    other_keys = [v for k, v in BFI_KEY_MAP.items() if k not in dims]
    interference = float(np.mean([abs(avg[k]) for k in other_keys]))

    result = {
        "label": args.label,
        "dims": dims,
        "alpha": alpha_simultaneous,
        "base_bfi": base_bfi,
        "simultaneous_avg": avg,
        "simultaneous_std": std,
        "simultaneous_ci95_half_width": ci95,
        "raw_runs": {k: [round(float(x), 4) for x in v] for k, v in delta_list.items()},
        "n_repeat": args.n_repeat,
        "interference_simultaneous": round(interference, 4),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = os.path.join(args.result_dir, f"three_vector_simultaneous_{dims_label}_{args.label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"간섭(나머지 {5-len(dims)}개 차원 평균 절대변화): {interference:.4f}")
    print(f"저장 완료 -> {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
