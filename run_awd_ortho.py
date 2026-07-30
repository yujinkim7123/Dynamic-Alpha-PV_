#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_awd_ortho.py
=====================================================================
프로젝트 원본 노트북(run_measure_matrix.ipynb 셀20)에서 그대로 가져온
AWD (Adaptive Weight Disentanglement) 직교화 스크립트입니다.
(Xiong et al., arXiv:2411.18729, 2024 / 논문 Eq. 9-11)

논문 수식:
    L = L_O + alpha * L_R                                    (Eq. 9)
    L_O = 1/(K(K-1)) * sum_{i!=j} |cos(phi_i - delta, phi_j - delta)|  (Eq. 10)
    L_R = ||delta||_2                                        (Eq. 11)
    delta^n = delta^{n-1} - beta * grad(L)   (Algorithm 1, beta=lr)

논문 하이퍼파라미터 (Section 4.3): alpha=0.1, beta=0.01, MAX_ITER=300

[ 실행 방법 ]
  python run_awd_ortho.py --phi_dir /path/to/phi_vectors \
                           --output_dir /path/to/ortho_phi_vectors \
                           --alpha 0.1 --lr 0.01 --max_iter 300

[ 저장 결과 ]
  {output_dir}/awd/
    phi_CON_High.pt   (merger.py 호환 파일명, phi_i - delta)
    phi_NEU_High.pt
    phi_EXT_High.pt
    phi_AGR_High.pt
    phi_OPN_High.pt

[ 시간이 매우 오래 걸립니다 ]
  300 iteration, 매 iteration마다 5C2=10쌍의 PASS1+PASS2 계산.
  중간에 끊기면 --resume_from 옵션으로 이어서 실행할 수 있습니다.
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

KEYS = ["CON_High", "NEU_High", "EXT_High", "AGR_High", "OPN_High"]


def os_free():
    gc.collect()
    torch.cuda.empty_cache()
    if libc:
        libc.malloc_trim(0)


def phi_path(key, phi_dir):
    return os.path.join(phi_dir, f"phi_{key}.pt")


def mem_usage():
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        free_v, total_v = torch.cuda.mem_get_info()
        used_v = (total_v - free_v) / 1024**3
        return f"RAM {rss:.1f}GB | VRAM {used_v:.1f}/{total_v/1024**3:.0f}GB"
    except Exception:
        return ""


def dict_to_flat(param_dict, sorted_keys):
    return torch.cat([param_dict[k].bfloat16().reshape(-1) for k in sorted_keys])


def compute_awd_loss(delta, phi_flats, alpha, device):
    K = len(phi_flats)
    eps = 1e-6
    n_pairs = K * (K - 1) // 2
    ij_list = [(i, j) for i in range(K) for j in range(i + 1, K)]

    grad_accum = torch.zeros(delta.numel(), dtype=torch.float32)  # 논문 Appendix A: float32 필수 (bfloat16은 gradient underflow로 학습이 거의 정체됨을 실전에서 확인)

    cos_sum = 0.0
    pair_info = []

    with torch.no_grad():
        for idx, (i, j) in enumerate(ij_list):
            logger.info(f"    PASS1 [{idx+1}/{n_pairs}] ({KEYS[i]}, {KEYS[j]}) | {mem_usage()}")

            phi_i = phi_flats[i].to(device)
            phi_j = phi_flats[j].to(device)
            delta_g = delta.to(device)

            vi = phi_i - delta_g
            vj = phi_j - delta_g

            ni = vi.norm().item()
            nj = vj.norm().item()
            dot = (vi * vj).sum().item()

            del phi_i, phi_j, delta_g, vi, vj
            os_free()

            if ni < eps or nj < eps:
                pair_info.append(None)
                continue

            cos = dot / (ni * nj + eps)
            cos_sum += 2 * abs(cos)   # 논문 Eq.10: 순서쌍 대칭성 반영 (|cos(i,j)|=|cos(j,i)|)
            pair_info.append((cos, ni, nj, dot))

        norm_delta = delta.to(device).norm().item()
        os_free()

    KK1 = K * (K - 1)  # = n_pairs * 2 = 20
    L_O = cos_sum / KK1   # 논문 Eq.10과 정확히 일치
    L_R = norm_delta
    L = L_O + alpha * L_R

    with torch.no_grad():
        for idx, info in enumerate(pair_info):
            if info is None:
                continue
            cos, ni, nj, dot = info
            i, j = ij_list[idx]
            sign = 1.0 if dot >= 0 else -1.0

            logger.info(f"    PASS2 [{idx+1}/{n_pairs}] ({KEYS[i]}, {KEYS[j]}) grad 계산 중...")

            phi_i = phi_flats[i].to(device)
            phi_j = phi_flats[j].to(device)
            delta_g = delta.to(device)

            vi = phi_i - delta_g
            vj = phi_j - delta_g
            del phi_i, phi_j
            os_free()

            # grad_ij를 항별로 순차 계산 (동시에 존재하는 큰 GPU 텐서 수를 최소화)
            denom_ij = ni * nj + eps
            acc = (vi + vj) * (sign / denom_ij)

            tmp = vi * (sign * cos / (ni ** 2 + eps))
            acc -= tmp
            del tmp
            os_free()

            tmp = vj * (sign * cos / (nj ** 2 + eps))
            acc -= tmp
            del tmp
            os_free()

            grad_ij = -acc / n_pairs
            del acc

            grad_accum += grad_ij.cpu()

            del delta_g, vi, vj, grad_ij
            os_free()

        if norm_delta > eps:
            grad_accum += (alpha * delta / (norm_delta + eps))

    return L, L_O, L_R, grad_accum


def optimize_delta(phi_flats, total_numel, alpha, lr, max_iter, tol, device,
                    delta_init=None, start_iter=0):
    delta = delta_init if delta_init is not None else torch.zeros(total_numel, dtype=torch.float32)  # float32 필수 (bfloat16은 gradient underflow로 학습 정체됨을 실전에서 확인)

    prev_loss = float("inf")
    logger.info(f"  delta: CPU bfloat16 {total_numel:,} | {mem_usage()}")
    logger.info(f"  Algorithm 1: delta^n = delta^(n-1) - beta*grad(L) | beta={lr} | alpha={alpha}")

    for it in range(start_iter, max_iter):
        t0 = time.time()
        logger.info(f"\n  -- iter [{it+1:4d}/{max_iter}] --------------------------")

        L, L_O, L_R, grad = compute_awd_loss(delta, phi_flats, alpha, device)
        loss_val = float(L)

        delta -= lr * grad
        del grad
        os_free()

        elapsed = time.time() - t0
        logger.info(
            f"  iter [{it+1:4d}/{max_iter}] done "
            f"loss={loss_val:.6f} L_O={L_O:.6f} norm_delta={L_R:.4f} | "
            f"{elapsed:.1f}s | {mem_usage()}"
        )

        if abs(prev_loss - loss_val) < tol and it > 10:
            logger.info(f"  수렴! iter={it+1}")
            break

        prev_loss = loss_val

    return delta


def measure_cosine(phi_flats, device, delta=None):
    K, eps, cos_list = len(phi_flats), 1e-8, []
    ij_list = [(i, j) for i in range(K) for j in range(i + 1, K)]

    with torch.no_grad():
        for i, j in ij_list:
            phi_i = phi_flats[i].to(device)
            phi_j = phi_flats[j].to(device)
            delta_g = delta.to(device) if delta is not None else None

            vi = phi_i - delta_g if delta_g is not None else phi_i
            vj = phi_j - delta_g if delta_g is not None else phi_j

            ni = vi.norm().item()
            nj = vj.norm().item()
            dot = (vi * vj).sum().item()

            del phi_i, phi_j, vi, vj
            if delta_g is not None:
                del delta_g
            os_free()

            if ni < eps or nj < eps:
                continue

            cos = abs(dot) / (ni * nj)
            cos_list.append(cos)
            logger.info(f"  |cos({KEYS[i]}, {KEYS[j]})| = {cos:.6f}  {'OK' if cos < 0.1 else 'WARN'}")

    return sum(cos_list) / len(cos_list) if cos_list else 0.0


def run(phi_dir, output_dir, alpha, lr, max_iter, tol, resume_from=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(output_dir, "awd")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  AWD (Adaptive Weight Disentanglement)")
    logger.info("  Xiong et al., arXiv:2411.18729, 2024")
    logger.info(f"  phi 폴더  : {phi_dir}")
    logger.info(f"  출력 폴더 : {out_dir}")
    logger.info(f"  대상      : {' | '.join(KEYS)}")
    logger.info(f"  alpha={alpha} | lr={lr} | max_iter={max_iter}")
    logger.info(f"  DEVICE={device} | phi+delta CPU, 쌍별 GPU 처리")
    logger.info("=" * 60)

    missing = [k for k in KEYS if not os.path.exists(phi_path(k, phi_dir))]
    if missing:
        logger.error(f"phi 파일 없음: {missing}")
        return None

    start = time.time()

    logger.info(f"\nSTEP 1: phi {len(KEYS)}개 로드 (CPU bfloat16)...")
    phi_flats = []
    sorted_keys = None
    ref_shapes = None

    for key in KEYS:
        logger.info(f"  [{key}] 로드 중...")
        phi_dict = torch.load(phi_path(key, phi_dir), map_location="cpu", weights_only=True)
        if sorted_keys is None:
            sorted_keys = sorted(phi_dict.keys())
            ref_shapes = {k: (v.shape, v.dtype) for k, v in phi_dict.items()}
        phi_flats.append(dict_to_flat(phi_dict, sorted_keys))
        del phi_dict
        gc.collect()
        logger.info(f"  [{key}] 완료 | {mem_usage()}")

    total_numel = phi_flats[0].numel()
    logger.info(f"  파라미터 수: {total_numel:,}")

    logger.info("\nSTEP 2: 직교화 전 코사인 유사도...")
    avg_before = measure_cosine(phi_flats, device, delta=None)
    logger.info(f"  평균 |cos|: {avg_before:.6f}")

    delta_init = None
    start_iter = 0
    if resume_from:
        logger.info(f"\n이전 delta에서 재개: {resume_from}")
        delta_init = torch.load(resume_from, map_location="cpu")

    logger.info(f"\nSTEP 3: delta 최적화 (Algorithm 1)...")
    delta = optimize_delta(phi_flats, total_numel, alpha, lr, max_iter, tol, device,
                            delta_init=delta_init, start_iter=start_iter)

    delta_save_path = os.path.join(out_dir, "_delta_checkpoint.pt")
    torch.save(delta, delta_save_path)
    logger.info(f"  delta 체크포인트 저장 (최종): {delta_save_path}")

    logger.info("\nSTEP 4: 직교화 후 코사인 유사도...")
    avg_after = measure_cosine(phi_flats, device, delta=delta)
    reduction = (avg_before - avg_after) / avg_before * 100 if avg_before > 0 else 0.0
    logger.info(f"  평균 |cos|: {avg_after:.6f}")
    logger.info(f"  간섭 감소: {reduction:.1f}%  {'OK' if reduction > 0 else 'WARN'}")

    logger.info("\nSTEP 5: phi_i - delta 저장...")
    saved_paths = {}

    for i, key in enumerate(KEYS):
        ortho_flat = phi_flats[i].float() - delta.float()

        result, offset = {}, 0
        for k in sorted_keys:
            shape, dtype = ref_shapes[k]
            numel = 1
            for s in shape:
                numel *= s
            result[k] = ortho_flat[offset:offset + numel].reshape(shape).to(dtype).clone()
            offset += numel

        del ortho_flat
        gc.collect()

        save_path = os.path.join(out_dir, f"phi_{key}.pt")
        torch.save(result, save_path)
        del result
        gc.collect()

        size_mb = os.path.getsize(save_path) / 1024**2
        logger.info(f"  [{key}] -> {save_path} ({size_mb:.1f} MB)")
        saved_paths[key] = save_path

    del phi_flats, delta
    os_free()

    elapsed = round(time.time() - start, 1)
    manifest = {
        "method": "AWD (Xiong et al., arXiv:2411.18729, 2024)",
        "phi_dir": phi_dir,
        "keys": KEYS,
        "alpha": alpha,
        "lr": lr,
        "max_iter": max_iter,
        "cos_before": round(avg_before, 6),
        "cos_after": round(avg_after, 6),
        "interference_reduction_pct": round(reduction, 2),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": elapsed,
        "files": saved_paths,
    }
    with open(os.path.join(output_dir, "manifest_awd.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"  AWD 완료! 소요: {elapsed}초")
    logger.info(f"  간섭 감소: {reduction:.1f}%")
    logger.info(f"{'='*60}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi_dir", required=True)
    parser.add_argument("--output_dir", default="./ortho_phi_vectors")
    parser.add_argument("--alpha", type=float, default=0.1, help="논문 기본값 0.1")
    parser.add_argument("--lr", type=float, default=0.01, help="논문 기본값 0.01 (beta)")
    parser.add_argument("--max_iter", type=int, default=300, help="논문 기본값 300")
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--resume_from", default=None,
                         help="이전 delta 체크포인트(_delta_checkpoint.pt)에서 재개")
    args = parser.parse_args()
    run(args.phi_dir, args.output_dir, args.alpha, args.lr, args.max_iter, args.tol,
        resume_from=args.resume_from)