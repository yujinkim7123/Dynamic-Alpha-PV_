#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_gs_ortho.py
=====================================================================
프로젝트 원본 노트북(run_measure_matrix.ipynb 셀10)에서 그대로 가져온
Gram-Schmidt 직교화 스크립트입니다. (고전적 GS: 원본 벡터 기준 투영 제거)

논문에 명시된 두 순서(OPN-first, CON-first)를 --order 인자로 선택합니다.

[ 실행 방법 ]
  python run_gs_ortho.py --phi_dir /path/to/phi_vectors \
                          --output_dir /path/to/ortho_phi_vectors \
                          --order opn_first
  python run_gs_ortho.py --phi_dir /path/to/phi_vectors \
                          --output_dir /path/to/ortho_phi_vectors \
                          --order con_first

[ 저장 결과 ]
  {output_dir}/gs_{order}/
    ortho_gs_OPN_High.pt  (또는 CON_High.pt 등, 처리 순서의 첫 벡터)
    ...
  ※ merger.py가 요구하는 phi_{key}.pt 형식으로 별도 리네임이 필요합니다
    (아래 rename_for_merger 함수가 자동으로 처리)
"""

import argparse
import gc
import json
import logging
import os
import shutil
import time

import torch

try:
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
except Exception:
    libc = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ORDER_MAP = {
    "opn_first": ["OPN_High", "CON_High", "EXT_High", "AGR_High", "NEU_High"],
    "con_first": ["CON_High", "NEU_High", "EXT_High", "AGR_High", "OPN_High"],  # 원본 노트북 기본값
}


def os_free():
    gc.collect()
    torch.cuda.empty_cache()
    if libc:
        libc.malloc_trim(0)


def dict_to_flat(param_dict: dict) -> torch.Tensor:
    """param dict -> 1D bfloat16 CPU tensor (pop으로 파괴하며 변환, RAM 절약)"""
    total_numel = sum(v.numel() for v in param_dict.values())
    flat = torch.empty(total_numel, dtype=torch.bfloat16, device="cpu")
    offset = 0
    for k in sorted(list(param_dict.keys())):
        v = param_dict.pop(k)
        numel = v.numel()
        flat[offset:offset + numel] = v.cpu().bfloat16().reshape(-1)
        offset += numel
        del v
    return flat


def phi_path(key, phi_dir):
    return os.path.join(phi_dir, f"phi_{key}.pt")


def load_phi(key, phi_dir):
    path = phi_path(key, phi_dir)
    assert os.path.exists(path), f"phi 파일 없음: {path}"
    return torch.load(path, map_location="cpu", weights_only=True)


def ram_usage():
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss / (1024 ** 3)
        vm = psutil.virtual_memory()
        return f"프로세스 {rss:.1f}GB | 시스템 여유 {vm.available/1024**3:.1f}GB"
    except ImportError:
        return "(psutil 없음)"


def gram_schmidt_step(key, phi_dir, prev_keys, gs_dir):
    """phi_{key}.pt 로드 -> 이전 직교 벡터들의 투영 제거 -> 저장"""
    phi_dict = load_phi(key, phi_dir)
    ref_shapes = {k: (v.shape, v.dtype) for k, v in phi_dict.items()}
    v_i = dict_to_flat(phi_dict)
    del phi_dict
    os_free()

    logger.info(f"  [{key}] GS 시작 | norm: {v_i.norm():.6f} | {ram_usage()}")

    for prev_key in prev_keys:
        prev_path = os.path.join(gs_dir, f"ortho_gs_{prev_key}.pt")
        if not os.path.exists(prev_path):
            logger.warning(f"  [{key}] 이전 파일 없음: {prev_path}")
            continue

        prev_dict = torch.load(prev_path, map_location="cpu", weights_only=True)
        u_j = dict_to_flat(prev_dict)
        del prev_dict

        dot_uu = torch.dot(u_j, u_j).item()
        if dot_uu < 1e-12:
            del u_j
            os_free()
            continue

        scalar = torch.dot(v_i, u_j).item() / dot_uu
        v_i.sub_(u_j, alpha=scalar)

        del u_j
        os_free()
        logger.info(f"  [{key}] -{prev_key} 투영 제거 | scalar={scalar:.6f} | {ram_usage()}")

    new_norm = v_i.norm().item()
    logger.info(f"  [{key}] GS 완료 | norm: {new_norm:.6f}")

    if new_norm < 1e-8:
        logger.warning(f"  [{key}] ⚠️  영벡터! 저장 건너뜀")
        del v_i
        os_free()
        return None

    result = {}
    offset = 0
    for k in sorted(ref_shapes.keys()):
        shape, dtype = ref_shapes[k]
        numel = 1
        for s in shape:
            numel *= s
        result[k] = v_i[offset:offset + numel].reshape(shape).to(dtype).clone()
        offset += numel

    del v_i
    os_free()
    logger.info(f"  [{key}] v_i 해제 후 | {ram_usage()}")

    save_path = os.path.join(gs_dir, f"ortho_gs_{key}.pt")
    torch.save(result, save_path)
    del result
    os_free()

    size_mb = os.path.getsize(save_path) / (1024 ** 2)
    logger.info(f"  [{key}] 저장 완료 → {save_path} ({size_mb:.1f} MB) | {ram_usage()}")
    return save_path


def rename_for_merger(gs_dir: str, keys: list):
    """ortho_gs_{key}.pt -> phi_{key}.pt 로 리네임 (merger.py 호환용, 복사)"""
    for key in keys:
        src = os.path.join(gs_dir, f"ortho_gs_{key}.pt")
        dst = os.path.join(gs_dir, f"phi_{key}.pt")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            logger.info(f"  리네임: {src} -> {dst}")


def run(phi_dir: str, output_dir: str, order_name: str):
    keys = ORDER_MAP[order_name]
    gs_dir = os.path.join(output_dir, f"gs_{order_name}")
    os.makedirs(gs_dir, exist_ok=True)

    logger.info("=" * 55)
    logger.info("  Gram-Schmidt 직교화 파이프라인")
    logger.info(f"  phi 경로  : {phi_dir}")
    logger.info(f"  출력 경로 : {gs_dir}")
    logger.info(f"  처리 순서 : {' -> '.join(keys)}")
    logger.info(f"  메모리 전략: 최대 동시 사용 ~30GB")
    logger.info("=" * 55)

    missing = [k for k in keys if not os.path.exists(phi_path(k, phi_dir))]
    if missing:
        logger.error(f"phi 파일 없음: {missing}")
        return None

    start = time.time()
    manifest = {
        "phi_dir": phi_dir,
        "gs_order": keys,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gs_files": {},
    }

    gs_saved_keys = []
    for i, key in enumerate(keys):
        logger.info(f"\n[{i+1}/{len(keys)}] == {key} ==")
        logger.info(f"  시작 전 RAM: {ram_usage()}")

        gs_path = gram_schmidt_step(key, phi_dir, gs_saved_keys, gs_dir)
        if gs_path is None:
            continue

        gs_saved_keys.append(key)
        manifest["gs_files"][key] = gs_path

        os_free()
        logger.info(f"  [{key}] 완료 ✅ | {ram_usage()}")

    manifest["elapsed_sec"] = round(time.time() - start, 1)
    manifest_path = os.path.join(output_dir, f"manifest_gs_{order_name}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # merger.py가 요구하는 phi_{key}.pt 형식으로 리네임
    logger.info("\nmerger.py 호환을 위해 파일명 정리 중...")
    rename_for_merger(gs_dir, gs_saved_keys)

    logger.info(f"\n{'='*55}")
    logger.info(f"  완료! 총 소요: {manifest['elapsed_sec']}초")
    logger.info(f"  저장: {len(manifest['gs_files'])}개 → {gs_dir}/")
    logger.info(f"{'='*55}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi_dir", required=True)
    parser.add_argument("--output_dir", default="./ortho_phi_vectors")
    parser.add_argument("--order", choices=["opn_first", "con_first"], required=True)
    args = parser.parse_args()
    run(args.phi_dir, args.output_dir, args.order)
