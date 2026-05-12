from __future__ import annotations

import os
from dataclasses import dataclass

from datasets import Dataset, DatasetDict, load_dataset

from personality_llm_pipeline.config import DatasetConfig


@dataclass
class PreparedDatasets:
    train: Dataset
    validation: Dataset


def load_training_dataset(config: DatasetConfig) -> DatasetDict:
    if config.source.endswith(".json") or config.source.endswith(".jsonl"):
        return load_dataset("json", data_files=config.source)

    token = os.environ.get("HF_TOKEN") if config.hf_use_auth_token else None
    return load_dataset(config.source, token=token)


def filter_dataset(dataset_dict: DatasetDict, config: DatasetConfig) -> DatasetDict:
    if config.trait_filter and config.level_filter:
        train_split = dataset_dict["train"].filter(
            lambda row: row[config.trait_column] == config.trait_filter
            and row[config.level_column] == config.level_filter
        )
        if "validation" in dataset_dict:
            val_split = dataset_dict["validation"].filter(
                lambda row: row[config.trait_column] == config.trait_filter
                and row[config.level_column] == config.level_filter
            )
        else:
            split = train_split.train_test_split(
                test_size=config.validation_split,
                seed=config.shuffle_seed,
            )
            return DatasetDict(train=split["train"], validation=split["test"])
        return DatasetDict(train=train_split, validation=val_split)

    if "validation" in dataset_dict:
        return dataset_dict

    split = dataset_dict["train"].train_test_split(
        test_size=config.validation_split,
        seed=config.shuffle_seed,
    )
    return DatasetDict(train=split["train"], validation=split["test"])


def validate_columns(example: dict, config: DatasetConfig) -> None:
    required = [config.instruction_column, config.output_column]
    missing = [column for column in required if column not in example]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
