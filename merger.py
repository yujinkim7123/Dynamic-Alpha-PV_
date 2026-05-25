

import os
import torch
import torch.nn as nn
import logging
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# 원본 코드 import (참조)
# =============================================================================
from task_vector import TaskVector                          # task_vector.py
from mask_weights_utils import mask_input_with_mask_rate   # mask_weights_utils.py


logger = logging.getLogger(__name__)


# =============================================================================
# 설정
# =============================================================================


PHI_DIR         = ""
BASE_MODEL_PATH = ""
ALPHA_THRESHOLD = 0.0


MANIFEST = {
    "OPN": {"High": "phi_OPN_High.pt", "Low": "phi_OPN_Low.pt"},
    "CON": {"High": "phi_CON_High.pt", "Low": "phi_CON_Low.pt"},
    "EXT": {"High": "phi_EXT_High.pt", "Low": "phi_EXT_Low.pt"},
    "AGR": {"High": "phi_AGR_High.pt", "Low": "phi_AGR_Low.pt"},
    "NEU": {"High": "phi_NEU_High.pt", "Low": "phi_NEU_Low.pt"},
}




# =============================================================================
# 원본 ties_merging 내부 헬퍼 함수들
# merging_methods.py ties_merging 안에 있던 것과 완전히 동일
# =============================================================================


def _task_vector_to_single_vector(task_vector: TaskVector) -> torch.Tensor:
\
    # deepcopy 하면 GPU 텐서가 CPU로 내려옴 → 제거
    sorted_tv_dict = OrderedDict(sorted(task_vector.task_vector_param_dict.items()))
    return nn.utils.parameters_to_vector(
        [param.flatten() for param in sorted_tv_dict.values()]
    )




def _single_vector_to_task_vector(
    single_vector: torch.Tensor,
    task_vector: TaskVector
) -> dict:
    """
    1D 벡터 → TaskVector param dict 변환.
    deepcopy 제거 → GPU 텐서 그대로 유지
    """
    # 구조만 참조 (deepcopy 불필요 — single_vector 로 값 덮어쓸 것이므로)
    sorted_keys = sorted(task_vector.task_vector_param_dict.keys())
    shapes      = [task_vector.task_vector_param_dict[k].shape for k in sorted_keys]


    result = {}
    offset = 0
    for key, shape in zip(sorted_keys, shapes):
        numel = 1
        for s in shape:
            numel *= s
        result[key] = single_vector[offset : offset + numel].reshape(shape).clone()
        offset += numel
    return result




def _mask_smallest_magnitude(
    flattened_params: torch.Tensor,
    param_value_mask_rate: float = 0.8
) -> torch.Tensor:
    """
    원본 merging_methods.py ties_merging 의
    mask_smallest_magnitude_param_values 와 완전히 동일.
    shape: (num_models, num_total_params)
    """
    num_mask = int(flattened_params.shape[1] * param_value_mask_rate)
    kth_values, _ = flattened_params.abs().kthvalue(
        k=num_mask, dim=1, keepdim=True
    )
    mask = flattened_params.abs() >= kth_values
    return flattened_params * mask




def _get_param_signs(flattened_params: torch.Tensor) -> torch.Tensor:
    """
    원본 merging_methods.py ties_merging 의
    get_param_signs 와 완전히 동일.
    """
    param_signs   = torch.sign(flattened_params.sum(dim=0))
    majority_sign = torch.sign(param_signs.sum(dim=0))
    param_signs[param_signs == 0] = majority_sign
    return param_signs




def _disjoint_merge(
    flattened_params: torch.Tensor,
    param_signs: torch.Tensor
) -> torch.Tensor:
    """
    원본 merging_methods.py ties_merging 의
    disjoint_merge 와 완전히 동일.
    """
    preserve_mask = (
        ((param_signs.unsqueeze(0) > 0) & (flattened_params > 0)) |
        ((param_signs.unsqueeze(0) < 0) & (flattened_params < 0))
    )
    preserved    = flattened_params * preserve_mask
    count        = (preserved != 0).sum(dim=0).float()
    merged_flat  = torch.sum(preserved, dim=0) / torch.clamp(count, min=1.0)
    return merged_flat




# =============================================================================
# DynamicMerger
# =============================================================================
class DynamicMerger:
    """
    φ 벡터(.pt) + base 모델을 이용한 실시간 동적 머징 클래스.


    Args:
        base_model_path  (str)  : base 모델 경로
        phi_dir          (str)  : phi .pt 파일 폴더
        method           (str)  : "task_arithmetic" | "dare" | "dare_ties"
        dare_drop_rate   (float): DaRE drop rate (Sun et al. 권장 0.5)
        dare_rescale     (bool) : DaRE rescale 여부 (권장 True)
        dare_strategy    (str)  : "random" | "magnitude"
        ties_trim_rate   (float): TIES magnitude trim rate (권장 0.8)
        scaling_coefficient (float): 최종 스케일 계수 (기본 1.0)
    """


    def __init__(
        self,
        base_model_path: str    = BASE_MODEL_PATH,
        phi_dir: str            = PHI_DIR,
        method: str             = "dare",
        dare_drop_rate: float   = 0.5,
        dare_rescale: bool      = True,
        dare_strategy: str      = "random",
        ties_trim_rate: float   = 0.8,
        scaling_coefficient: float = 1.0,
    ):
        self.phi_dir             = phi_dir
        self.method              = method
        self.dare_drop_rate      = dare_drop_rate
        self.dare_rescale        = dare_rescale
        self.dare_strategy       = dare_strategy
        self.ties_trim_rate      = ties_trim_rate
        self.scaling_coefficient = scaling_coefficient


        # ── GPU 디바이스 설정 ─────────────────────────────────────────────────
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[Merger] 사용 디바이스: {self.device}")


        # ── base 모델을 GPU에 로드 ────────────────────────────────────────────
        # base_model 만 GPU 상주 (14GB)
        # 나머지(base_params, φ캐시)는 CPU pinned memory에 보관
        # → CPU→GPU 전송 속도 2~3배 향상 (pinned memory = page-locked)
        logger.info(f"[Merger] base 모델 로드 (GPU): {base_model_path}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,  #=torch.bfloat16,
            device_map=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)


        # ── θ_base 복사본 → GPU 보관 ─────────────────────────────────────────
        # VRAM 96GB 기준:
        #   base_model(14GB) + base_params(14GB) + φ 1개(14GB) = 42GB → 여유
        # GPU→GPU copy 가 CPU pinned→GPU 보다 빠름 → base_params는 GPU에 유지
        logger.info(f"[Merger] θ_base 복사본 → GPU 보관")
        self.base_params = {
            name: param.data.clone()   # GPU에 그대로 복사
            for name, param in self.base_model.named_parameters()
        }
        logger.info(f"[Merger] θ_base GPU 보관 완료")


        # ── φ 벡터 10개 전부 CPU RAM에 캐싱 ─────────────────────────────────
        # RAM 1.7TB 기준: φ 10개 × 14GB = 140GB → 전혀 문제없음
        # merge() 호출 시 캐시에서 GPU로 이동 → 디스크 I/O 완전 제거
        logger.info(f"[Merger] φ 벡터 캐싱 시작 (10개 → CPU RAM)")
        self.phi_cache = {}
        for dim in MANIFEST:
            # for direction in ["High", "Low"]:
            for direction in ["High"]:
                key      = f"{dim}_{direction}"
                filename = MANIFEST[dim][direction]
                filepath = os.path.join(phi_dir, filename)
                if not os.path.exists(filepath):
                    logger.warning(f"  [φ 캐시] 파일 없음, 건너뜀: {filepath}")
                    continue
                phi_dict = torch.load(filepath, map_location="cpu", weights_only=False)
                phi_dict = {k: v.to(torch.bfloat16) for k, v in phi_dict.items()}
                self.phi_cache[key] = TaskVector(task_vector_param_dict=phi_dict)
                logger.info(f"  [φ 캐시] {key} ✅")
        logger.info(f"[Merger] φ 캐시 완료 — {len(self.phi_cache)}개 CPU RAM 상주")
        logger.info(f"[Merger] 초기화 완료 — method={method}")




    # ──────────────────────────────────────────────────────────────────────────
    # φ 로드 → TaskVector 로 래핑
    # ──────────────────────────────────────────────────────────────────────────
    def _load_phi(self, dim: str, direction: str) -> TaskVector:
        """
        CPU RAM 캐시 → GPU TaskVector 반환.
        디스크 I/O 없음 → 속도 향상
        """
        key = f"{dim}_{direction}"
        if key not in self.phi_cache:
            raise KeyError(f"φ 캐시에 없음: {key}")


        # CPU RAM → GPU 이동
        phi_gpu = {
            k: v.to(self.device)
            for k, v in self.phi_cache[key].task_vector_param_dict.items()
        }
        return TaskVector(task_vector_param_dict=phi_gpu)




    # ──────────────────────────────────────────────────────────────────────────
    # DaRE 적용
    # 원본 mask_merging 의 mask_input_with_mask_rate 루프와 동일
    # ──────────────────────────────────────────────────────────────────────────
    def _apply_dare(self, tv: TaskVector) -> TaskVector:
        """
        원본 mask_merging 브랜치:
        각 파라미터에 mask_input_with_mask_rate 적용.
        mask_weights_utils.py 의 mask_input_with_mask_rate 직접 호출.
        """
        dare_dict = {}
        with torch.no_grad():
            for name, tensor in tv.task_vector_param_dict.items():
                dare_dict[name] = mask_input_with_mask_rate(
                    input_tensor  = tensor,
                    mask_rate     = self.dare_drop_rate,
                    use_rescale   = self.dare_rescale,
                    mask_strategy = self.dare_strategy,
                )
        return TaskVector(task_vector_param_dict=dare_dict)




    # ──────────────────────────────────────────────────────────────────────────
    # TIES 3단계 적용
    # 원본 merging_methods.py ties_merging 과 완전히 동일한 로직
    # 차이점: φ 가 이미 α 스케일된 상태로 들어옴 (모델 로드 불필요)
    # ──────────────────────────────────────────────────────────────────────────
    def _apply_ties_full(self, task_vectors: list) -> TaskVector:
        """
        원본 ties_merging 3단계 로직 그대로.


        Args:
            task_vectors: [TaskVector, ...] — 활성 차원별 α 스케일된 φ 리스트
                          ※ 모든 TaskVector의 텐서는 CPU에 있어야 함


        Returns:
            TaskVector — TIES 3단계 적용된 최종 φ (텐서는 CPU)


        ── TIES 연산을 GPU에서 수행 ────────────────────────────────────────
        φ 벡터가 GPU 캐시에 상주하므로 vstack / sign / disjoint merge
        모두 GPU에서 처리됨 → CPU 연산 대비 속도 대폭 향상
        최대 VRAM: φ 5개 동시 vstack → 활성 차원 수 × ~1.4GB ≈ 7GB 추가
        ─────────────────────────────────────────────────────────────────────
        """
        # φ 벡터들을 1D로 펼쳐서 쌓기 — CPU에서 수행
        # shape: (활성 차원 수, 전체 파라미터 수)
        flattened_list = [
            _task_vector_to_single_vector(tv) for tv in task_vectors
        ]
        # [수정] vstack이 CPU 텐서끼리 연산 → GPU 점유 없음
        flattened_params = torch.vstack(flattened_list)


        with torch.no_grad():
            # 1단계: magnitude trim — CPU 연산
            flattened_params = _mask_smallest_magnitude(
                flattened_params      = flattened_params,
                param_value_mask_rate = self.ties_trim_rate,
            )


            # 2단계: 부호 합의 — CPU 연산
            # sign/sum/compare 연산이므로 CPU에서도 빠르게 처리됨
            param_signs = _get_param_signs(flattened_params)


            # 3단계: disjoint merge — CPU 연산
            merged_flat = _disjoint_merge(flattened_params, param_signs)


            # 1D → TaskVector param dict 복원 — CPU 텐서 그대로 반환
            # merge() 에서 .to(param.device) 로 GPU 이동
            merged_tv_dict = _single_vector_to_task_vector(
                single_vector = merged_flat,
                task_vector   = task_vectors[0],   # 구조 참조용
            )


        return TaskVector(task_vector_param_dict=merged_tv_dict)




    # ──────────────────────────────────────────────────────────────────────────
    # 메인 머징
    # 수식 7: θ' = θ_base + Σ α_i × φ_i
    # ──────────────────────────────────────────────────────────────────────────
    def merge(self, alpha: dict) -> AutoModelForCausalLM:
        """
        Args:
            alpha: {dim: α값} — dynamic_alpha.process_turn() 출력


        Returns:
            머징된 base_model
        """
        # 활성 차원 필터링
        active_dims = {
            dim: val for dim, val in alpha.items()
            if abs(val) > ALPHA_THRESHOLD
        }


        if not active_dims:
            logger.info("[Merger] 활성 차원 없음 → 머징 건너뜀")
            return self.base_model


        logger.info(f"[Merger] 머징 시작 — method={self.method}")
        logger.info(f"  활성 차원: { {k: round(v,4) for k,v in active_dims.items()} }")


        with torch.no_grad():


            # ── θ_base 복원 ──────────────────────────────────────────────────
            # base_params 가 GPU에 있으므로 GPU→GPU copy → 가장 빠름
            for name, param in self.base_model.named_parameters():
                param.data.copy_(self.base_params[name])


            # ================================================================
            # task_arithmetic
            # φ 캐시(GPU) → α 스케일(GPU) → 누적(GPU) → base_model 반영
            # ================================================================
            if self.method == "task_arithmetic":


                merged_tv = None
                for dim, alpha_val in active_dims.items():
                    direction = "High" if alpha_val > 0 else "Low"
                    abs_alpha = abs(alpha_val)


                    # φ 캐시에서 즉시 참조 (디스크 I/O 없음)
                    tv = self._load_phi(dim, direction)


                    if merged_tv is None:
                        merged_tv = abs_alpha * tv
                    else:
                        merged_tv = merged_tv + (abs_alpha * tv)


                    logger.info(f"  [{dim}] α={alpha_val:+.4f} ({direction}) ✅")


                merged_params = merged_tv.combine_with_pretrained_model(
                    pretrained_model    = self.base_model,
                    scaling_coefficient = 1.0,
                )
                for name, param in self.base_model.named_parameters():
                    if name in merged_params:
                        param.data.copy_(merged_params[name])


                del merged_tv
                torch.cuda.empty_cache()


            # ================================================================
            # dare
            # φ 캐시(GPU) → DaRE(GPU) → aggregated_delta(GPU) 누적 → 반영
            # ================================================================
            elif self.method == "dare":


                # aggregated_delta GPU에서 초기화
                aggregated_delta = {
                    name: torch.zeros_like(param)   # GPU에 생성
                    for name, param in self.base_model.named_parameters()
                }


                for dim, alpha_val in active_dims.items():
                    direction = "High" if alpha_val > 0 else "Low"
                    abs_alpha = abs(alpha_val)


                    # φ 캐시(GPU) → DaRE(GPU)
                    tv      = self._load_phi(dim, direction)
                    tv_dare = self._apply_dare(tv)


                    # aggregated_delta 누적 — GPU 연산
                    for name in aggregated_delta:
                        if name in tv_dare.task_vector_param_dict:
                            aggregated_delta[name].add_(
                                tv_dare.task_vector_param_dict[name] * abs_alpha
                            )


                    del tv_dare
                    torch.cuda.empty_cache()
                    logger.info(f"  [{dim}] DaRE α={alpha_val:+.4f} ({direction}) ✅")


                # GPU aggregated_delta → base_model에 직접 반영
                for name, param in self.base_model.named_parameters():
                    param.data.add_(
                        aggregated_delta[name] * self.scaling_coefficient
                    )


                del aggregated_delta
                torch.cuda.empty_cache()


            # ================================================================
            # dare_ties
            # φ 캐시(GPU) → DaRE(GPU) → α 스케일(GPU)
            # → TIES 3단계(GPU) → base_model 반영
            # ================================================================
            elif self.method == "dare_ties":


                # STEP 1: φ 캐시(GPU) → DaRE(GPU) → α 스케일(GPU)
                scaled_tvs = []
                for dim, alpha_val in active_dims.items():
                    direction = "High" if alpha_val > 0 else "Low"
                    abs_alpha = abs(alpha_val)


                    tv      = self._load_phi(dim, direction)
                    tv_dare = self._apply_dare(tv)
                    tv_scaled = abs_alpha * tv_dare
                    scaled_tvs.append(tv_scaled)


                    del tv_dare
                    torch.cuda.empty_cache()
                    logger.info(f"  [{dim}] DaRE 완료 α={alpha_val:+.4f} ({direction})")


                # STEP 2: TIES 3단계 — GPU에서 수행
                # φ 5개가 이미 GPU 캐시에 있으므로 vstack도 GPU에서 처리
                merged_tv = self._apply_ties_full(scaled_tvs)
                del scaled_tvs
                torch.cuda.empty_cache()


                # STEP 3: base_model에 반영
                merged_params = merged_tv.combine_with_pretrained_model(
                    pretrained_model    = self.base_model,
                    scaling_coefficient = self.scaling_coefficient,
                )
                for name, param in self.base_model.named_parameters():
                    if name in merged_params:
                        param.data.copy_(merged_params[name])


                del merged_tv
                torch.cuda.empty_cache()


            else:
                raise ValueError(f"지원하지 않는 method: {self.method}")


        logger.info("[Merger] 머징 완료 ✅")
        return self.base_model




# =============================================================================
# 단독 실행 테스트
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")


    merger = DynamicMerger(
        base_model_path     = BASE_MODEL_PATH,
        phi_dir             = PHI_DIR,
        method              = "dare",   # "task_arithmetic" | "dare" | "dare_ties"
        dare_drop_rate      = 0.5,
        dare_rescale        = True,
        dare_strategy       = "random",
        ties_trim_rate      = 0.8,
        scaling_coefficient = 1.0,
    )


    test_alpha = {
        "OPN":  0.00,
        "CON": +0.10,
        "EXT":  0.00,
        "AGR": +0.10,
        "NEU":  0.00,
    }


    print(f"\n입력 alpha: {test_alpha}")
    merged_model = merger.merge(test_alpha)
    print("✅ 머징 완료!")


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# merger_cpu.py  ★ GPU 상주 / CPU 연산 버전 (디버그 로그 포함)
# """


# import gc
# import os
# import psutil
# import torch
# import torch.nn as nn
# import logging
# from collections import OrderedDict
# from transformers import AutoModelForCausalLM, AutoTokenizer


# from task_vector import TaskVector
# from mask_weights_utils import mask_input_with_mask_rate


# logger = logging.getLogger(__name__)


# PHI_DIR         = "/workspace/auto_personality/phi_vectors"
# BASE_MODEL_PATH = "/workspace/auto_personality/qwen_base"
# ALPHA_THRESHOLD = 0.0


# MANIFEST = {
#     "OPN": {"High": "phi_OPN_High.pt", "Low": "phi_OPN_Low.pt"},
#     "CON": {"High": "phi_CON_High.pt", "Low": "phi_CON_Low.pt"},
#     "EXT": {"High": "phi_EXT_High.pt", "Low": "phi_EXT_Low.pt"},
#     "AGR": {"High": "phi_AGR_High.pt", "Low": "phi_AGR_Low.pt"},
#     "NEU": {"High": "phi_NEU_High.pt", "Low": "phi_NEU_Low.pt"},
# }




# # =============================================================================
# # 메모리 상태 출력 헬퍼
# # =============================================================================
# def _mem(tag: str) -> None:
#     """현재 RAM / VRAM 사용량을 GB 단위로 출력."""
#     ram_gb = psutil.Process().memory_info().rss / 1024**3
#     ram_avail_gb = psutil.virtual_memory().available / 1024**3


#     if torch.cuda.is_available():
#         vram_used_gb  = torch.cuda.memory_allocated() / 1024**3
#         vram_reserved_gb = torch.cuda.memory_reserved() / 1024**3
#         logger.info(
#             f"  [MEM] {tag} | "
#             f"RAM used={ram_gb:.1f}GB avail={ram_avail_gb:.1f}GB | "
#             f"VRAM alloc={vram_used_gb:.1f}GB reserved={vram_reserved_gb:.1f}GB"
#         )
#     else:
#         logger.info(
#             f"  [MEM] {tag} | "
#             f"RAM used={ram_gb:.1f}GB avail={ram_avail_gb:.1f}GB"
#         )




# # =============================================================================
# # TIES 헬퍼 함수 — 원본과 100% 동일
# # =============================================================================


# def _task_vector_to_single_vector(task_vector: TaskVector) -> torch.Tensor:
#     sorted_tv_dict = OrderedDict(sorted(task_vector.task_vector_param_dict.items()))
#     return nn.utils.parameters_to_vector(
#         [param.flatten() for param in sorted_tv_dict.values()]
#     )




# def _single_vector_to_task_vector(
#     single_vector: torch.Tensor,
#     task_vector: TaskVector,
# ) -> dict:
#     sorted_keys = sorted(task_vector.task_vector_param_dict.keys())
#     shapes      = [task_vector.task_vector_param_dict[k].shape for k in sorted_keys]
#     result = {}
#     offset = 0
#     for key, shape in zip(sorted_keys, shapes):
#         numel = 1
#         for s in shape:
#             numel *= s
#         result[key] = single_vector[offset : offset + numel].reshape(shape).clone()
#         offset += numel
#     return result




# def _mask_smallest_magnitude(
#     flattened_params: torch.Tensor,
#     param_value_mask_rate: float = 0.8,
# ) -> torch.Tensor:
#     """
#     원본과 동일한 수식: abs >= kth_value 인 원소만 유지.
#     행별 처리 + in-place 수정으로 중간 임시 텐서 최소화.
#     flattened_params 를 직접 in-place 수정해서 반환 → 별도 result 텐서 불필요.
#     """
#     num_mask = int(flattened_params.shape[1] * param_value_mask_rate)


#     for i in range(flattened_params.shape[0]):
#         row     = flattened_params[i]       # view — 복사 없음
#         abs_row = row.abs()                 # 14.2 GB
#         kth_val, _ = abs_row.kthvalue(k=num_mask)
#         mask_row = abs_row >= kth_val
#         del abs_row, kth_val                # 즉시 해제


#         # in-place 로 row (= flattened_params[i] view) 에 직접 곱함
#         # → 별도 임시 텐서 없음
#         row.mul_(mask_row)
#         del mask_row                        # 즉시 해제


#     # flattened_params 자체가 in-place 수정됨 → 그대로 반환
#     return flattened_params




# def _get_param_signs(flattened_params: torch.Tensor) -> torch.Tensor:
#     """원본과 동일: sum → sign → majority_sign → fill zeros."""
#     col_sum     = flattened_params.sum(dim=0)
#     param_signs = torch.sign(col_sum)
#     del col_sum                             # 즉시 해제
#     majority_sign = torch.sign(param_signs.sum(dim=0))
#     param_signs[param_signs == 0] = majority_sign
#     return param_signs




# def _disjoint_merge(
#     flattened_params: torch.Tensor,
#     param_signs: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     원본과 동일한 수식: 부호 일치 원소만 보존 후 평균.
#     행별 처리로 중간 bool 텐서 (21GB × 6개) 동시 생성 방지.
#     """
#     num_cols  = flattened_params.shape[1]
#     sum_buf   = torch.zeros(num_cols, dtype=flattened_params.dtype)
#     count_buf = torch.zeros(num_cols, dtype=torch.float32)


#     for i in range(flattened_params.shape[0]):
#         row = flattened_params[i]           # view — 복사 없음


#         # 1D bool 마스크 — 작음
#         keep = ((param_signs > 0) & (row > 0)) | ((param_signs < 0) & (row < 0))


#         preserved_row = row * keep
#         del keep                            # 즉시 해제


#         sum_buf.add_(preserved_row)


#         nonzero = (preserved_row != 0)     # bool 1D
#         count_buf.add_(nonzero.float())
#         del nonzero, preserved_row          # 즉시 해제


#     merged_flat = sum_buf / torch.clamp(count_buf, min=1.0)
#     del sum_buf, count_buf                  # 즉시 해제
#     return merged_flat




# # =============================================================================
# # DynamicMerger
# # =============================================================================
# class DynamicMerger:


#     def __init__(
#         self,
#         base_model_path: str       = BASE_MODEL_PATH,
#         phi_dir: str               = PHI_DIR,
#         method: str                = "dare",
#         dare_drop_rate: float      = 0.5,
#         dare_rescale: bool         = True,
#         dare_strategy: str         = "random",
#         ties_trim_rate: float      = 0.7,
#         scaling_coefficient: float = 1.0,
#     ):
#         self.phi_dir             = phi_dir
#         self.method              = method
#         self.dare_drop_rate      = dare_drop_rate
#         self.dare_rescale        = dare_rescale
#         self.dare_strategy       = dare_strategy
#         self.ties_trim_rate      = ties_trim_rate
#         self.scaling_coefficient = scaling_coefficient


#         self.gpu = "cuda" if torch.cuda.is_available() else "cpu"
#         logger.info(f"[Merger] base_model → {self.gpu} / φ 연산 → cpu")


#         _mem("init 시작")


#         logger.info(f"[Merger] base_model 로드: {base_model_path}")
#         self.base_model = AutoModelForCausalLM.from_pretrained(
#             base_model_path,
#             torch_dtype=torch.bfloat16,
#             device_map=self.gpu,
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
#         _mem("base_model 로드 완료")


#         logger.info("[Merger] θ_base CPU 복사본 저장 중...")
#         self.base_params = {
#             name: param.data.clone().cpu()
#             for name, param in self.base_model.named_parameters()
#         }
#         _mem("base_params 저장 완료")
#         logger.info(f"[Merger] 초기화 완료 — method={method}")




#     def _load_phi(self, dim: str, direction: str) -> TaskVector:
#         filename = MANIFEST[dim][direction]
#         filepath = os.path.join(self.phi_dir, filename)
#         if not os.path.exists(filepath):
#             raise FileNotFoundError(f"φ 파일 없음: {filepath}")


#         logger.info(f"  [φ] {dim}_{direction} 디스크 → CPU 읽는 중...")
#         phi_dict = torch.load(filepath, map_location="cpu", weights_only=False)
#         phi_dict = {k: v.to(torch.bfloat16) for k, v in phi_dict.items()}
#         tv = TaskVector(task_vector_param_dict=phi_dict)
#         _mem(f"φ {dim}_{direction} 로드 완료")
#         return tv




#     def _apply_dare(self, tv: TaskVector) -> TaskVector:
#             dare_dict = {}
#             # 현재 시스템의 GPU 디바이스 설정 (cuda)
#             device = self.gpu 
            
#             with torch.no_grad():
#                 for name, tensor in tv.task_vector_param_dict.items():
#                     # 1. 연산을 위해 GPU로 전송
#                     gpu_tensor = tensor.to(device)
                    
#                     # 2. GPU에서 mask 연산 수행
#                     masked_gpu_tensor = mask_input_with_mask_rate(
#                         input_tensor  = gpu_tensor,
#                         mask_rate     = self.dare_drop_rate,
#                         use_rescale   = self.dare_rescale,
#                         mask_strategy = self.dare_strategy,
#                     )
                    
#                     # 3. 결과를 다시 CPU로 복사 및 메모리 해제
#                     dare_dict[name] = masked_gpu_tensor.cpu()
                    
#                     del gpu_tensor, masked_gpu_tensor
                    
#                 # 루프 종료 후 GPU 캐시 정리 (선택 사항)
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
                    
#             return TaskVector(task_vector_param_dict=dare_dict)




#     def _apply_ties_full(self, task_vectors: list) -> TaskVector:
#         logger.info("  [TIES] flatten 시작...")
#         _mem("TIES flatten 전")


#         flattened_list = [_task_vector_to_single_vector(tv) for tv in task_vectors]
#         _mem("TIES flatten 완료")


#         logger.info("  [TIES] vstack 시작...")


#         with torch.no_grad():
#             # vstack: (활성 차원 수, 전체 파라미터 수) shape CPU 텐서
#             flattened_params = torch.vstack(flattened_list)


#             # ★ flattened_list 즉시 해제 — vstack 이 독립 텐서를 만들었으므로 불필요
#             del flattened_list
#             gc.collect()
#             _mem("TIES vstack 완료 / flattened_list 해제 후")


#             # ★ 실제 크기 확인
#             logger.info(f"  [TIES] flattened_params shape: {flattened_params.shape}")
#             logger.info(f"  [TIES] flattened_params size:  {flattened_params.nbytes / 1024**3:.2f} GB")
#             logger.info(f"  [TIES] flattened_params dtype: {flattened_params.dtype}")


#             logger.info("  [TIES] magnitude trim 중...")
#             # _mask_smallest_magnitude 가 flattened_params 를 in-place 수정 후 반환
#             # → trimmed 중간 변수 불필요, 42GB 추가 복사 없음
#             flattened_params = _mask_smallest_magnitude(
#                 flattened_params      = flattened_params,
#                 param_value_mask_rate = self.ties_trim_rate,
#             )
#             gc.collect()
#             _mem("TIES trim 완료")


#             logger.info("  [TIES] sign 합의 중...")
#             param_signs = _get_param_signs(flattened_params)
#             _mem("TIES sign 완료")


#             logger.info("  [TIES] disjoint merge 중...")
#             merged_flat = _disjoint_merge(flattened_params, param_signs)


#             # ★ 더 이상 필요없는 텐서 즉시 해제
#             del flattened_params, param_signs
#             gc.collect()
#             _mem("TIES disjoint 완료 / 중간 텐서 해제 후")


#             merged_tv_dict = _single_vector_to_task_vector(
#                 single_vector = merged_flat,
#                 task_vector   = task_vectors[0],   # 구조 참조용
#             )


#             # ★ merged_flat 즉시 해제
#             del merged_flat
#             gc.collect()
#             _mem("TIES 벡터 복원 완료 / merged_flat 해제 후")


#         return TaskVector(task_vector_param_dict=merged_tv_dict)




#     def _write_to_gpu(self, delta_dict: dict, scaling: float) -> None:
#         """파라미터 1개씩 CPU 계산 후 GPU param 에 copy_() — 임시 텐서 없음."""
#         total = len(list(self.base_model.named_parameters()))
#         logger.info(f"  [write] GPU 반영 시작 — 파라미터 {total}개")
#         _mem("write_to_gpu 시작")


#         for i, (name, param) in enumerate(self.base_model.named_parameters()):
#             if name in delta_dict:
#                 cpu_val = self.base_params[name] + scaling * delta_dict[name]
#             else:
#                 cpu_val = self.base_params[name]


#             param.data.copy_(cpu_val)
#             del cpu_val


#             # 100개마다 메모리 상태 출력
#             if (i + 1) % 100 == 0:
#                 _mem(f"write {i+1}/{total}")


#         _mem("write_to_gpu 완료")
#         logger.info("  [write] GPU 반영 완료 ✅")




#     def merge(self, alpha: dict) -> AutoModelForCausalLM:
#         active_dims = {
#             dim: val for dim, val in alpha.items()
#             if abs(val) > ALPHA_THRESHOLD
#         }


#         if not active_dims:
#             logger.info("[Merger] 활성 차원 없음 → 건너뜀")
#             return self.base_model


#         logger.info(f"[Merger] 머징 시작 — method={self.method}")
#         logger.info(f"  활성 차원: { {k: round(v,4) for k,v in active_dims.items()} }")
#         _mem("merge() 시작")


#         with torch.no_grad():


#             # ================================================================
#             # task_arithmetic
#             # ================================================================
#             if self.method == "task_arithmetic":


#                 merged_tv = None


#                 for dim, alpha_val in active_dims.items():
#                     direction = "High" if alpha_val > 0 else "Low"
#                     abs_alpha = abs(alpha_val)


#                     tv = self._load_phi(dim, direction)


#                     if merged_tv is None:
#                         merged_tv = abs_alpha * tv
#                     else:
#                         merged_tv = merged_tv + (abs_alpha * tv)


#                     del tv
#                     gc.collect()
#                     _mem(f"task_arithmetic [{dim}] 누적 후")
#                     logger.info(f"  [{dim}] α={alpha_val:+.4f} ({direction}) ✅")


#                 _mem("task_arithmetic write_to_gpu 직전")
#                 self._write_to_gpu(merged_tv.task_vector_param_dict, scaling=1.0)


#                 del merged_tv
#                 gc.collect()
#                 _mem("task_arithmetic 완료")


#             # ================================================================
#             # dare
#             # ================================================================
#             elif self.method == "dare":


#                 aggregated_delta = {
#                     name: torch.zeros_like(self.base_params[name])
#                     for name in self.base_params
#                 }
#                 _mem("dare aggregated_delta 초기화 완료")


#                 for dim, alpha_val in active_dims.items():
#                     direction = "High" if alpha_val > 0 else "Low"
#                     abs_alpha = abs(alpha_val)


#                     tv      = self._load_phi(dim, direction)
#                     tv_dare = self._apply_dare(tv)


#                     del tv
#                     gc.collect()
#                     _mem(f"dare [{dim}] φ 해제 후")


#                     for name in aggregated_delta:
#                         if name in tv_dare.task_vector_param_dict:
#                             aggregated_delta[name].add_(
#                                 tv_dare.task_vector_param_dict[name] * abs_alpha
#                             )


#                     del tv_dare
#                     gc.collect()
#                     _mem(f"dare [{dim}] DaRE 해제 후")
#                     logger.info(f"  [{dim}] DaRE α={alpha_val:+.4f} ({direction}) ✅")


#                 _mem("dare write_to_gpu 직전")
#                 self._write_to_gpu(aggregated_delta, scaling=self.scaling_coefficient)


#                 del aggregated_delta
#                 gc.collect()
#                 _mem("dare 완료")


#             # ================================================================
#             # dare_ties
#             # ================================================================
#             elif self.method == "dare_ties":


#                 # STEP 1
#                 scaled_tvs = []
#                 _mem("dare_ties STEP1 시작")


#                 for dim, alpha_val in active_dims.items():
#                     direction = "High" if alpha_val > 0 else "Low"
#                     abs_alpha = abs(alpha_val)


#                     logger.info(f"  [{dim}] φ 로드 중...")
#                     tv      = self._load_phi(dim, direction)


#                     logger.info(f"  [{dim}] DaRE 적용 중...")
#                     tv_dare = self._apply_dare(tv)


#                     del tv
#                     gc.collect()
#                     _mem(f"dare_ties [{dim}] φ 해제 후")


#                     tv_scaled = abs_alpha * tv_dare
#                     scaled_tvs.append(tv_scaled)


#                     del tv_dare
#                     gc.collect()
#                     _mem(f"dare_ties [{dim}] DaRE 해제 후 / scaled_tvs={len(scaled_tvs)}개")
#                     logger.info(f"  [{dim}] DaRE 완료 α={alpha_val:+.4f} ({direction})")


#                 # STEP 2
#                 logger.info("  [STEP2] TIES 3단계 시작...")
#                 _mem("dare_ties STEP2 TIES 시작 전")
#                 merged_tv = self._apply_ties_full(scaled_tvs)


#                 logger.info("  [STEP2] scaled_tvs 해제 중...")
#                 del scaled_tvs
#                 gc.collect()
#                 _mem("dare_ties scaled_tvs 해제 후")


#                 # STEP 3
#                 logger.info("  [STEP3] GPU 반영 시작...")
#                 _mem("dare_ties STEP3 write_to_gpu 직전")
#                 self._write_to_gpu(
#                     merged_tv.task_vector_param_dict,
#                     scaling=self.scaling_coefficient,
#                 )


#                 logger.info("  [STEP3] merged_tv 해제 중...")
#                 del merged_tv
#                 gc.collect()
#                 _mem("dare_ties 완료")


#             else:
#                 raise ValueError(f"지원하지 않는 method: {self.method}")


#         logger.info("[Merger] 머징 완료 ✅")
#         _mem("merge() 종료")
#         return self.base_model




# # =============================================================================
# # 단독 실행 테스트
# # =============================================================================
# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s %(message)s",
#         datefmt="%H:%M:%S",
#     )


#     merger = DynamicMerger(
#         base_model_path     = BASE_MODEL_PATH,
#         phi_dir             = PHI_DIR,
#         method              = "dare_ties",
#         dare_drop_rate      = 0.5,
#         dare_rescale        = True,
#         dare_strategy       = "random",
#         ties_trim_rate      = 0.8,
#         scaling_coefficient = 1.0,
#     )


#     test_alpha = {
#         "OPN":  0.00,
#         "CON": +0.10,
#         "EXT":  0.00,
#         "AGR": +0.10,
#         "NEU":  0.00,
#     }


#     print(f"\n입력 alpha: {test_alpha}")
#     merged_model = merger.merge(test_alpha)
#     print("✅ 머징 완료!")

    def _reset_to_base(self):
        """base_params로 모델 파라미터 복원 (반복 실험용)"""
        import torch
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                param.data.copy_(self.base_params[name])
        logger.info("[Merger] 베이스 복원 완료")
