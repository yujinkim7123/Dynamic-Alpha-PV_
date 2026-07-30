#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_mgs_ortho.py
=====================================================================
Modified Gram-Schmidt (MGS) 직교화 스크립트.
GS 원본 코드(run_gs_ortho.py)와 동일한 메모리 관리 전략을 따르되,
알고리즘만 "매 단계 갱신된 벡터 기준"으로 순차 처리하도록 변경했습니다.

논문 Eq.7 (GS)과 수학적으로 동치이지만, 8B 파라미터 규모에서
수치적으로 더 안정적인 재정식화입니다 (논문 4.1절 명시).

논문에 명시된 두 순서(Table 5, 6과 동일하게 MGS에도 적용):
    OPN-first: OPN -> CON -> EXT -> AGR -> NEU
    CON-first: CON -> NEU -> EXT -> AGR -> OPN

[ 실행 방법 ]
    python run_mgs_ortho.py --phi_dir /path/to/phi_vectors \
                             --output_dir /path/to/ortho_phi_vectors \
                             --order opn_first
    python run_mgs_ortho.py --phi_dir /path/to/phi_vectors \
                             --output_dir /path/to/ortho_phi_vectors \
                             --order con_first

[ 저장 결과 ]
    {output_dir}/mgs_{order}/
        phi_OPN_High.pt   ← merger.py 호환 파일명 (바로 사용 가능)
        phi_CON_High.pt
        ...
"""

import argparse
import ctypes
import gc
import json
import logging
import os
import time

import torch

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

ORDER_MAP = {
    "opn_first": ["OPN_High", "CON_High", "EXT_High", "AGR_High", "NEU_High"],
    "con_first": ["CON_High", "NEU_High", "EXT_High", "AGR_High", "OPN_High"],
}


def os_free():
    gc.collect()
    torch.cuda.empty_cache()
    if libc:
        libc.malloc_trim(0)


def dict_to_flat(param_dict: dict) -> torch.Tensor:
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


def unflatten_to_dict(flat: torch.Tensor, ref_shapes: dict) -> dict:
    result = {}
    offset = 0
    for k in sorted(ref_shapes.keys()):
        shape, dtype = ref_shapes[k]
        numel = 1
        for s in shape:
            numel *= s
        result[k] = flat[offset:offset + numel].reshape(shape).to(dtype).clone()
        offset += numel
    return result


def run(phi_dir: str, output_dir: str, order_name: str):
    keys = ORDER_MAP[order_name]
    mgs_dir = os.path.join(output_dir, f"mgs_{order_name}")
    os.makedirs(mgs_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Modified Gram-Schmidt (MGS) 직교화 파이프라인")
    logger.info(f"  phi 경로  : {phi_dir}")
    logger.info(f"  출력 경로 : {mgs_dir}")
    logger.info(f"  처리 순서 : {' -> '.join(keys)}")
    logger.info(f"  메모리 전략: 5개 벡터 동시 로드 (~75GB RAM)")
    logger.info("=" * 60)

    missing = [k for k in keys if not os.path.exists(phi_path(k, phi_dir))]
    if missing:
        logger.error(f"phi 파일 없음: {missing}")
        return None

    start = time.time()

    logger.info(f"\nSTEP 1: phi {len(keys)}개 로드 및 flatten...")
    ref_shapes = None
    vectors = {}

    for key in keys:
        logger.info(f"  [{key}] 로드 중... | {ram_usage()}")
        phi_dict = load_phi(key, phi_dir)
        if ref_shapes is None:
            ref_shapes = {k: (v.shape, v.dtype) for k, v in phi_dict.items()}
        vectors[key] = dict_to_flat(phi_dict).float()
        del phi_dict
        os_free()
        logger.info(f"  [{key}] 완료 | norm={vectors[key].norm().item():.6f} | {ram_usage()}")

    logger.info("\nSTEP 2: MGS 순차 직교화 (처리 즉시 저장, 메모리 누적 방지)...")

    saved_paths = {}

    for i, key_i in enumerate(keys):
        v_i = vectors[key_i]
        norm_i = v_i.norm().item()
        logger.info(f"\n  [{i+1}/{len(keys)}] 기준: {key_i} | norm={norm_i:.6f}")

        if norm_i < 1e-8:
            logger.warning(f"  [{key_i}] 영벡터! 그대로 저장")

        # ── 이 시점의 v_i가 곧 최종 직교화 결과 -> 즉시 저장 (clone 없이) ──
        result = unflatten_to_dict(v_i, ref_shapes)
        save_path = os.path.join(mgs_dir, f"phi_{key_i}.pt")
        torch.save(result, save_path)
        del result
        os_free()
        size_mb = os.path.getsize(save_path) / (1024 ** 2)
        logger.info(f"  [{key_i}] 저장 완료 -> {save_path} ({size_mb:.1f} MB)")
        saved_paths[key_i] = save_path

        if norm_i < 1e-8:
            del vectors[key_i]
            os_free()
            continue

        u_i = v_i / norm_i

        # 남은 모든 벡터에서 u_i 성분 제거 (MGS 핵심: 이미 갱신된 v_j를 계속 갱신)
        for key_j in keys[i + 1:]:
            v_j = vectors[key_j]
            coeff = torch.dot(v_j, u_i).item()
            v_j.sub_(u_i, alpha=coeff)
            logger.info(f"    -> {key_j}에서 {key_i} 성분 제거 | coeff={coeff:.6f} | new_norm={v_j.norm().item():.6f}")

        del u_i
        # ── 처리 끝난 v_i를 vectors에서 즉시 제거 (메모리 누적 방지 핵심) ──
        del vectors[key_i]
        os_free()
        logger.info(f"  [{key_i}] 완료, 메모리에서 해제 | {ram_usage()}")

    del vectors
    os_free()

    logger.info("\nSTEP 4: 직교성 검증 (샘플 파라미터 20개)...")
    _verify(keys, mgs_dir)

    elapsed = round(time.time() - start, 1)
    manifest = {
        "method": "Modified Gram-Schmidt (MGS)",
        "order": order_name,
        "phi_dir": phi_dir,
        "keys": keys,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": elapsed,
        "files": saved_paths,
    }
    with open(os.path.join(output_dir, f"manifest_mgs_{order_name}.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"  MGS ({order_name}) 완료! 소요: {elapsed}초")
    logger.info(f"  저장: {len(saved_paths)}개 -> {mgs_dir}/")
    logger.info(f"{'='*60}")

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
    parser.add_argument("--order", choices=["opn_first", "con_first"], required=True)
    args = parser.parse_args()
    run(args.phi_dir, args.output_dir, args.order)