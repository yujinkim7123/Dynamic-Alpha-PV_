from __future__ import annotations

import argparse
import torch

from personality_llm_pipeline.train import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train with the original fullfinetuning.py interface.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--data-path", dest="data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--lr_scheduler", default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--train_on_inputs", default=False)
    parser.add_argument("--add_eos_token", default=False)
    parser.add_argument("--group_by_length", default=False)
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--prompt_template_name", default="alpaca")
    parser.add_argument("--trait", default=None)
    parser.add_argument("--level", default=None)
    return parser


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def main() -> None:
    args = build_parser().parse_args()
    torch.cuda.empty_cache()
    train(
        base_model=args.base_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        lr_scheduler=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        train_on_inputs=_to_bool(args.train_on_inputs),
        add_eos_token=_to_bool(args.add_eos_token),
        group_by_length=_to_bool(args.group_by_length),
        wandb_run_name=args.wandb_run_name,
        resume_from_checkpoint=args.resume_from_checkpoint,
        prompt_template_name=args.prompt_template_name,
        trait=args.trait,
        level=args.level,
    )


if __name__ == "__main__":
    main()
