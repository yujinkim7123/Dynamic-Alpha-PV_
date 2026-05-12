#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import argparse
import os
import sys
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 논문 깃허브에서 제공하는 모듈들(동일 디렉토리/적절한 PYTHONPATH 가정)
from model_merging_methods.merging_methods import MergingMethod
from utils.utils import set_random_seed

def load_model_and_tokenizer(model_path: str, device="cpu"):
    """
    Hugging Face transformers로 AutoModelForCausalLM + AutoTokenizer 로딩
    """
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser("Interface for Merging LLMs (no inference)")

    # 필수 인자
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Base model checkpoint path")
    parser.add_argument("--finetuned_model_paths", type=str, nargs="+", required=True,
                        help="One or multiple fine-tuned model paths")
    parser.add_argument("--output_model_path", type=str, required=True,
                        help="Where to save the merged model")

    # merging_method_name
    parser.add_argument("--merging_method_name", type=str, default="average_merging",
                        choices=["average_merging", "task_arithmetic", "mask_merging",
                                 "fisher_merging", "regmean_merging", "ties_merging"],
                        help="Top-level merging method (see MergingMethod in merging_methods.py)")

    # mask_apply_method: mask_merging일 때 DaRE 후 어떻게 병합할지
    parser.add_argument("--mask_apply_method", type=str, default="average_merging",
                        choices=["average_merging", "task_arithmetic", "fisher_merging",
                                 "regmean_merging", "ties_merging"],
                        help="When merging_method_name=mask_merging, how to combine masked models")

    # DaRE (mask_merging) 관련
    parser.add_argument("--weight_format", type=str, default="delta_weight",
                        choices=["delta_weight", "finetuned_weight"],
                        help="DaRE 시 마스킹할 파라미터 형식 (delta_weight or finetuned_weight)")
    parser.add_argument("--weight_mask_rate", type=float, default=0.0,
                        help="Drop rate (p)")
    parser.add_argument("--use_weight_rescale", action="store_true", default=False,
                        help="Whether to rescale param by 1/(1-p) after dropping")
    parser.add_argument("--mask_strategy", type=str, default="random",
                        choices=["random", "magnitude"],
                        help="DARE 마스킹 전략")

    # Task Arithmetic 등에서 스케일 파라미터
    parser.add_argument("--scaling_coefficient", type=float, default=1.0,
                        help="Scaling factor for task_arithmetic or TIES, etc.")

    # Fisher, RegMean 등은 trainer 기반 추가 인자가 필요할 수 있지만 여기서는 생략

    # exclude params regex
    parser.add_argument("--exclude_param_names_regex", type=str, nargs="*", default=[],
                        help="정규표현식으로 제외할 파라미터")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--individual_scaling", type=str, default="false",
                    choices=["true", "false"],
                    help="If true, apply individual scaling coefficients for each model in task_arithmetic merging.")
    parser.add_argument("--scaling_coefficients", type=float, nargs="+", default=None,
                        help="Individual scaling coefficients for each fine-tuned model if individual_scaling is true.")

    args = parser.parse_args()

    individual_scaling = args.individual_scaling.lower() == "true"

    if individual_scaling:
        if args.scaling_coefficients is None or len(args.scaling_coefficients) != len(args.finetuned_model_paths):
            raise ValueError("When using individual_scaling, scaling_coefficients must be provided and match the number of finetuned models.")


    # 로거 세팅
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.info(f"Running with args={args}")

    # 시드 고정
    set_random_seed(args.seed)

    # 1) Base 모델 로딩
    logger.info(f"Loading base model from {args.base_model_path}")
    base_model, base_tokenizer = load_model_and_tokenizer(args.base_model_path, device="cpu")

    # 2) Fine-tuned 모델들 로딩
    ft_models = []
    for ft_path in args.finetuned_model_paths:
        logger.info(f"Loading fine-tuned model from {ft_path}")
        ft_m, _ = load_model_and_tokenizer(ft_path, device="cpu")
        ft_models.append(ft_m)

    # 3) MergingMethod 생성
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)

    logger.info(f"Number of fine-tuned models: {len(ft_models)}")

    # 4) 병합 (논문 깃허브의 "get_merged_model()" 사용)
    #    => 내부적으로 Delta 계산/마스킹(DARE)/TIES 등 로직이 처리됨
    #       trainers, fisher 등은 None or 기본값
    merged_model = merging_method.get_merged_model(
        merged_model=base_model,
        models_to_merge=ft_models,
        exclude_param_names_regex=args.exclude_param_names_regex,
        trainers=[None]*len(ft_models),
        scaling_coefficient=args.scaling_coefficient,
        individual_scaling=individual_scaling,
        scaling_coefficients=args.scaling_coefficients,
        nums_fisher_examples=None,
        fisher_scaling_coefficients=None,
        normalize_fisher_weight=True,
        minimal_fisher_weight=1e-6,
        nums_regmean_examples=None,
        reduce_non_diagonal_ratio=1.0,
        param_value_mask_rate=0.8,  # TIES 등에서 쓸 수도 있음
        weight_format=args.weight_format,
        weight_mask_rates=[args.weight_mask_rate]*len(ft_models),
        use_weight_rescale=args.use_weight_rescale,
        mask_strategy=args.mask_strategy,
        mask_apply_method=args.mask_apply_method,
        models_use_deepcopy=False
    )

    # 5) 최종 모델 저장
    logger.info(f"Saving merged model to {args.output_model_path}")
    os.makedirs(args.output_model_path, exist_ok=True)
    merged_model.save_pretrained(args.output_model_path)

    # tokenizer: base tokenizer 그대로 저장 (일반적으로 vocab 변화 없음 가정)
    base_tokenizer.save_pretrained(args.output_model_path)

    logger.info("Merging finished. No inference was performed.")

if __name__ == "__main__":
    main()
