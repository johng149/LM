from torch.utils.data import DataLoader
from src.common.models.dataloader_type import DataloaderType
from src.datasets.base.processor import Processor
from datasets import load_dataset, load_from_disk
from typing import List, Callable, Any, Tuple, Optional, Union
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from src.common.models.verification import Verification
from src.common.services.verification import verify_args
from src.common.models.args_info import ArgInfo
from src.common.models.param_level import ParamLevel
from src.datasets.utils.masking import (
    causal_self_attn_mask,
    self_attn_pad_mask,
    process_tokens,
    combine_masks_before_flip,
)


class WMT19EnZhProcessor(Processor):
    def __init__(self, info):
        super().__init__(info)

    def encode(self, sample):
        sample = sample["translation"]
        en = sample["en"]
        zh = sample["zh"]
        en_enc = self.tokenizer.encode(en, return_tensors="pt").squeeze(0)
        zh_enc = self.tokenizer.encode(zh, return_tensors="pt").squeeze(0)
        return {"en": en, "en_encoded": en_enc, "zh": zh, "zh_encoded": zh_enc}

    def process_helper(
        self,
        save_path: str,
        format: str | None,
        max_en_len: int,
        max_zh_len: int,
        approx_len_mult: int = 2,
        **kwargs
    ):
        wmt19_en_zh = load_dataset("wmt19", "zh-en")
        train_data = wmt19_en_zh["train"]
        val_data = wmt19_en_zh["validation"]

        def short_approx_filter(example):
            return (
                len(example["translation"]["en"]) < approx_len_mult * max_en_len
                and len(example["translation"]["zh"]) < approx_len_mult * max_zh_len
            )

        train_data = train_data.filter(short_approx_filter)
        val_data = val_data.filter(short_approx_filter)

        train_data = train_data.map(self.encode, remove_columns=["translation"])
        val_data = val_data.map(self.encode, remove_columns=["translation"])

        def short_filter(example):
            return (
                len(example["en_encoded"]) <= max_en_len
                and len(example["zh_encoded"]) <= max_zh_len
            )

        train_data = train_data.filter(short_filter)
        val_data = val_data.filter(short_filter)

        if format:
            train_data = train_data.with_format(format)
            val_data = val_data.with_format(format)

        train_path = self.format_dataset_path(save_path, DataloaderType.TRAIN)
        val_path = self.format_dataset_path(save_path, DataloaderType.VALIDATION)

        train_data.save_to_disk(train_path)
        val_data.save_to_disk(val_path)
