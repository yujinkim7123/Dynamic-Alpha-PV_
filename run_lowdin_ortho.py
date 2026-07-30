#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_lowdin_ortho.py
=====================================================================
프로젝트 원본 노트북(run_measure_matrix.ipynb 셀17)에서 그대로 가져온
Löwdin (Symmetric Orthogonalization) 스크립트입니다.
논문 Eq.8: Phi_orth = Phi * S^(-1/2),  S = Phi^T Phi

[ 실행 방법 ]
  python run_lowdin_ortho.py --phi_dir /path/to/phi_vectors --output_dir /path/to/ortho_phi_vectors

[ 저장 결과 ]
  {output_dir}/lowdin/
    phi_CON_High.pt   ← merger.py 호환 파일명
    phi_NEU_High.pt
    phi_EXT_High.pt
    phi_AGR_High.pt
    phi_OPN_High.pt
"""

import argparse
import ctypes
import gc
import json
import logging
import os
import time

import torch

torch.set_num_threads(4)

try:
    libc = ctypes.CDLL("libc.so.6")
except Exception:
    libc = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

KEYS = ["CON_High", "NEU_High", "EXT_High", "AGR_High", "OPN_High"]
CHUNK_SIZE = 10_000_000
EPS = 1e-12


def os_free():
    gc.collect()
    torch.cuda.empty_cache()
    if libc:
        libc.malloc_trim(0)


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
        return f"RSS {rss:.1f}GB | 여유 {vm.available/1024**3:.1f}GB"
    except ImportError:
        return ""


def run(phi_dir, output_dir, keys=None):
    keys = keys or KEYS
    out_dir = os.path.join(output_dir, "lowdin")
    tmp_dir = os.path.join(out_dir, "_tmp")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    n = len(keys)

    logger.info("=" * 60)
    logger.info("  Löwdin (Symmetric Orthogonalization)")
    logger.info(f"  phi 폴더  : {phi_dir}")
    logger.info(f"  출력 폴더 : {out_dir}")
    logger.info(f"  대상      : {' | '.join(keys)}")
    logger.info("=" * 60)

    missing = [k for k in keys if not os.path.exists(phi_path(k, phi_dir))]
    if missing:
        logger.error(f"phi 파일 없음: {missing}")
        return None

    start = time.time()

    logger.info(f"\nSTEP 1: phi {n}개 로드 중... (총 ~{n*15}GB 예상)")
    phis = {}
    for key in keys:
        logger.info(f"  [{key}] 로드 중...")
        phis[key] = load_phi(key, phi_dir)
        logger.info(f"  [{key}] 완료 | {ram_usage()}")

    sorted_param_keys = sorted(phis[keys[0]].keys())
    total_params = sum(phis[keys[0]][k].numel() for k in sorted_param_keys)
    logger.info(f"\n  총 파라미터: {total_params:,}")

    chunks, cur, cur_size = [], [], 0
    for pk in sorted_param_keys:
        numel = phis[keys[0]][pk].numel()
        cur.append((pk, numel))
        cur_size += numel
        if cur_size >= CHUNK_SIZE:
            chunks.append(cur)
            cur, cur_size = [], 0
    if cur:
        chunks.append(cur)

    ref_shapes = {k: (v.shape, v.dtype) for k, v in phis[keys[0]].items()}
    logger.info(f"  청크 수: {len(chunks)} (청크당 {CHUNK_SIZE//1_000_000}M 파라미터)")

    logger.info("\nSTEP 2: Gram 행렬 S = V^T V 계산...")
    S = torch.zeros(n, n, dtype=torch.float32)
    log_interval = max(1, len(chunks) // 10)

    for chunk_idx, chunk_params in enumerate(chunks):
        chunk_param_keys = [pk for pk, _ in chunk_params]
        cols = [
            torch.cat([phis[key][pk].float().reshape(-1) for pk in chunk_param_keys])
            for key in keys
        ]
        V_chunk = torch.stack(cols, dim=1)
        del cols
        S += V_chunk.T @ V_chunk
        del V_chunk
        gc.collect()
        if (chunk_idx + 1) % log_interval == 0:
            logger.info(f"  [{chunk_idx+1}/{len(chunks)}] S 누적 중...")

    logger.info(f"  S 완료:\n{S}")

    logger.info("\nSTEP 3: S^{-1/2} 계산...")
    eigenvalues, Q = torch.linalg.eigh(S)
    logger.info(f"  eigenvalues: {eigenvalues.tolist()}")
    eigenvalues_safe = torch.clamp(eigenvalues.abs(), min=EPS)
    S_inv_sqrt = Q @ torch.diag(eigenvalues_safe ** (-0.5)) @ Q.T
    del S, eigenvalues, eigenvalues_safe, Q
    gc.collect()
    logger.info("  S^{-1/2} 완료")

    logger.info("\nSTEP 4: U = V S^{-1/2} 계산 → 임시 저장...")
    for chunk_idx, chunk_params in enumerate(chunks):
        chunk_param_keys = [pk for pk, _ in chunk_params]
        cols = [
            torch.cat([phis[key][pk].float().reshape(-1) for pk in chunk_param_keys])
            for key in keys
        ]
        V_chunk = torch.stack(cols, dim=1).float()
        del cols
        U_chunk = V_chunk @ S_inv_sqrt
        del V_chunk
        gc.collect()

        for vec_idx, key in enumerate(keys):
            tmp_path = os.path.join(tmp_dir, f"tmp_{key}_{chunk_idx:05d}.pt")
            torch.save(U_chunk[:, vec_idx].clone().bfloat16(), tmp_path)
        del U_chunk
        gc.collect()

        if (chunk_idx + 1) % log_interval == 0:
            logger.info(f"  [{chunk_idx+1}/{len(chunks)}] U 계산 중... {ram_usage()}")

    del S_inv_sqrt
    gc.collect()

    logger.info("\nSTEP 5: phi 메모리 해제...")
    for key in keys:
        del phis[key]
    del phis
    os_free()
    logger.info(f"  phi 해제 완료 | {ram_usage()}")

    logger.info("\nSTEP 6: 임시 청크 합쳐서 최종 저장...")
    saved_paths = {}
    for key in keys:
        chunk_tensors = []
        for chunk_idx in range(len(chunks)):
            tmp_path = os.path.join(tmp_dir, f"tmp_{key}_{chunk_idx:05d}.pt")
            chunk_tensors.append(torch.load(tmp_path, map_location="cpu", weights_only=True))
            os.remove(tmp_path)

        full_flat = torch.cat(chunk_tensors)
        del chunk_tensors
        gc.collect()

        result, offset = {}, 0
        for k in sorted(ref_shapes.keys()):
            shape, dtype = ref_shapes[k]
            numel = 1
            for s in shape:
                numel *= s
            result[k] = full_flat[offset:offset+numel].reshape(shape).to(dtype).clone()
            offset += numel

        del full_flat
        os_free()

        save_path = os.path.join(out_dir, f"phi_{key}.pt")
        torch.save(result, save_path)
        del result
        os_free()

        size_mb = os.path.getsize(save_path) / (1024 ** 2)
        logger.info(f"  [{key}] → {save_path} ({size_mb:.1f} MB) | {ram_usage()}")
        saved_paths[key] = save_path

    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    logger.info("\nSTEP 7: 직교성 검증 (샘플 파라미터 20개)...")
    _verify(keys, out_dir)

    elapsed = round(time.time() - start, 1)
    manifest = {
        "method": "Löwdin (Symmetric Orthogonalization)",
        "phi_dir": phi_dir,
        "keys": keys,
        "chunk_size": CHUNK_SIZE,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": elapsed,
        "files": saved_paths,
    }
    with open(os.path.join(output_dir, "manifest_lowdin.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Löwdin 완료! 소요: {elapsed}초")
    logger.info(f"  저장: {len(saved_paths)}개 → {out_dir}/")
    logger.info(f"{'='*60}")
    logger.info(f"\n[ merger 연결 방법 ]  phi_dir = '{out_dir}'")

    return manifest


def _verify(keys, out_dir):
    first = torch.load(os.path.join(out_dir, f"phi_{keys[0]}.pt"), map_location="cpu", weights_only=True)
    sample_keys = sorted(first.keys())[:20]
    del first
    os_free()

    vecs = {}
    for key in keys:
        phi = torch.load(os.path.join(out_dir, f"phi_{key}.pt"), map_location="cpu", weights_only=True)
        vecs[key] = torch.cat([phi[k].float().reshape(-1) for k in sample_keys])
        del phi
        os_free()

    cos_list = []
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            n1 = vecs[k1].norm().item()
            n2 = vecs[k2].norm().item()
            if n1 < 1e-12 or n2 < 1e-12:
                continue
            cos = torch.dot(vecs[k1], vecs[k2]).item() / (n1 * n2)
            ok = "✅" if abs(cos) < 0.05 else "⚠️ "
            logger.info(f"  cos({k1}, {k2}) = {cos:.8f}  {ok}")
            cos_list.append(abs(cos))

    if cos_list:
        logger.info(f"  평균 |cos|: {sum(cos_list)/len(cos_list):.8f}")

    for key in keys:
        del vecs[key]
    os_free()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi_dir", required=True)
    parser.add_argument("--output_dir", default="./ortho_phi_vectors")
    args = parser.parse_args()
    run(args.phi_dir, args.output_dir)
