from __future__ import annotations

import os
import random
import sys
import time

import torch
import transformers
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from personality_llm_pipeline.personality_tests_module import args as personality_args
from personality_llm_pipeline.personality_tests_module import personality_assessment
from personality_llm_pipeline.prompting import PromptFormatter


debug_samples = 1
counter = 0


class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):  # type: ignore[override]
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_folder)
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


class PersonalityEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, base_model_path: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.base_model_path = base_model_path

    def on_evaluate(self, args, state, control, **kwargs):  # type: ignore[override]
        model = kwargs["model"]
        temp_dir = os.path.join(args.output_dir, f"temp_eval_{int(time.time())}_{random.randint(0,9999)}")
        os.makedirs(temp_dir, exist_ok=True)
        model.save_pretrained(temp_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(temp_dir)
        else:
            print("[WARN] No tokenizer provided to PersonalityEvalCallback.")

        personality_args.llama_model_path = temp_dir
        personality_args.agent_type = "mycustom"
        personality_args.system_prompt = "Imagine you are a real person..."
        personality_args.questionnaire_name = "BFI"
        personality_args.character = "myagent"
        personality_args.agent_llm = "gpt-3.5-turbo"
        personality_args.evaluator_llm = "gpt-4o"
        personality_args.eval_method = "interview_batch"

        personality_assessment(
            character=personality_args.character,
            agent_type=personality_args.agent_type,
            agent_llm=personality_args.agent_llm,
            questionnaire_name=personality_args.questionnaire_name,
            eval_method=personality_args.eval_method,
            evaluator_llm=personality_args.evaluator_llm,
        )
        return control


def train(
    base_model: str = "",
    data_path: str = "",
    output_dir: str = "",
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    lr_scheduler: str = "cosine",
    warmup_steps: int = 100,
    train_on_inputs: bool = False,
    add_eos_token: bool = False,
    group_by_length: bool = False,
    wandb_run_name: str = "",
    resume_from_checkpoint: str | None = None,
    prompt_template_name: str = "alpaca",
    trait: str | None = None,
    level: str | None = None,
):
    global counter
    counter = 0

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Params using prompt template {prompt_template_name}:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_steps: {warmup_steps}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"trait: {trait}\n"
            f"level: {level}\n"
        )

    assert base_model, "Please specify --base_model='huggyllama/llama-7b' or similar"

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    formatter = PromptFormatter(
        template_name=prompt_template_name,
        system_prompt="Imagine you are a real person rather than a language model.",
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    print(type(model), model)
    print("length of tokenizer:", len(tokenizer))

    # LLaMA : pad_token_id 가 None 이므로 eos 로 대체
    # Qwen  : pad_token_id (<|endoftext|>) 가 이미 존재하므로 그대로 유지
    #         eos (<|im_end|>) 로 덮어쓰면 문장 끝과 패딩을 구분 못해 학습 불안정
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print(f"Pad token id: {tokenizer.pad_token_id}")
    print(f"eos token id: {tokenizer.eos_token_id}")

    def tokenize(prompt, add_eos=True):
        res = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            res["input_ids"][-1] != tokenizer.eos_token_id
            and len(res["input_ids"]) < cutoff_len
            and add_eos
        ):
            res["input_ids"].append(tokenizer.eos_token_id)
            res["attention_mask"].append(1)
        res["labels"] = res["input_ids"].copy()
        return res

    def generate_and_tokenize_prompt(data_point):
        global counter
        counter += 1

        full_prompt = formatter.format(
            data_point["train_instruction"],
            data_point.get("train_input", ""),
            data_point["train_output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        if not train_on_inputs:
            user_prompt = formatter.format(
                data_point["train_instruction"],
                data_point["train_input"],
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos=add_eos_token)
            # LLaMA: BOS(1개) + 헤더(5개) = 6개 → N - 6
            # Qwen : BOS 없음 + 헤더(3개) = 3개 → N - 3
            if prompt_template_name in {"llama", "llama_chat"}:
                header_offset = 6
            elif prompt_template_name == "qwen":
                header_offset = 3
            else:
                header_offset = 6
            user_len = len(tokenized_user_prompt["input_ids"]) - header_offset
            if add_eos_token:
                user_len -= 1
            tokenized_full_prompt["labels"] = [-100] * user_len + tokenized_full_prompt["labels"][user_len:]

        if counter <= debug_samples:
            print("=== SAMPLE ===")
            print("Full prompt:", full_prompt)
            print("Tokenized:", tokenized_full_prompt)

        return tokenized_full_prompt

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        print("=== private dataset ===")
        data = load_dataset(data_path, token=True)

    if trait and level:
        data = data.filter(lambda x: x["trait"] == trait and x["level"] == level)

    if "validation" in data:
        val_data = data["validation"].shuffle().map(generate_and_tokenize_prompt)
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    else:
        splitted = data["train"].train_test_split(test_size=0.1)
        train_data = splitted["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = splitted["test"].shuffle().map(generate_and_tokenize_prompt)

    print(f"Filtered: {len(train_data)} train samples, {len(val_data)} val samples")

    if resume_from_checkpoint:
        ckpt = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(resume_from_checkpoint, "adapter_model.bin")
            resume_from_checkpoint = True
        if os.path.exists(ckpt):
            print("Restart from:", ckpt)
            torch.load(ckpt)
        else:
            print("Checkpoint not found:", ckpt)

    remove_cols = [
        "env_idx",
        "trait",
        "level",
        "head",
        "relation",
        "tail",
        "literal",
        "narrative",
        "personx",
        "persony",
        "original_index",
    ]
    existing_remove_cols = [column for column in remove_cols if column in train_data.column_names]
    train_data = train_data.remove_columns(existing_remove_cols)
    if val_data:
        val_data = val_data.remove_columns(existing_remove_cols)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=batch_size,
            #warmup_ratio=0.06,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            remove_unused_columns=True,
            dataloader_num_workers=4,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            weight_decay=0.01,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=2000,
            save_steps=2000,
            lr_scheduler_type=lr_scheduler,
            output_dir=output_dir,
            gradient_checkpointing=True,
            load_best_model_at_end=True,
            #group_by_length=group_by_length,
            report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
            run_name=wandb_run_name,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
        callbacks=[
            SavePeftModelCallback(),
            # PersonalityEvalCallback(
            #     tokenizer=tokenizer,
            #     base_model_path="",
            # ),
        ],
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    for batch in trainer.get_train_dataloader():
        print("Debug first batch input_ids:", batch["input_ids"])
        break

    print("=== Training Start ===")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print("=== Saving final ===")

    tokenizer.save_pretrained(output_dir)
    trainer.save_model(output_dir)
    model.save_pretrained(output_dir)

    pt_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save({}, pt_path)