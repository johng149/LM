from src.tokenizers.models.info import Info
from src.common.models.verification import Verification
from src.common.models.args_info import ArgsInfo
from src.common.services.verification import verify_args
from torch.utils.data import DataLoader
from typing import Optional
import json
import os
from pathlib import Path


class Processor:

    def __init__(self, info: Info):
        self.info = info
        self.tokenizer = info.tokenizer
        self.pad_idx = info.pad_idx
        self.bos_idx = info.bos_idx
        self.eos_idx = info.eos_idx
        self.mask_idx = info.mask_idx
        self.vocab_size = info.vocab_size

    def process(self, ignore_cache: bool = False, **kwargs):
        raise NotImplementedError

    def already_cached(self, path: str, class_name: str, **kwargs) -> bool:
        # we look for a json file at the specified path, we expect
        # that the file is named after the class_name and that it
        # contains the kwargs as a dictionary
        expected_path = Path(path) / f"{class_name}.json"
        if not expected_path.exists():
            return False
        with open(expected_path, "r") as f:
            expected_kwargs = json.load(f)
        return expected_kwargs == kwargs

    def validate_args(self, **kwargs) -> Verification:
        return verify_args({}, **kwargs)

    def encode(self, sample):
        raise NotImplementedError

    def seq2seq(self) -> Optional[DataLoader]:
        return None

    def supports_seq2seq(self) -> bool:
        return False

    def causal(self) -> Optional[DataLoader]:
        return None

    def supports_causal(self) -> bool:
        return False
