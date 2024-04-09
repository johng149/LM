from torch.utils.data import DataLoader
from src.common.models.dataloader_type import DataloaderType
from src.datasets.base.processor import Processor
from datasets import load_dataset, load_from_disk
from typing import List, Callable, Any
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from src.common.models.verification import Verification
from src.common.services.verification import verify_args
from src.common.models.args_info import ArgInfo
from src.common.models.param_level import ParamLevel


class TinyShakespeareProcessor(Processor):

    def __init__(self, info):
        super().__init__(info)

    def encode(self, sample):
        source = sample["text"]
        enc = self.tokenizer.encode(source, return_tensors="pt").squeeze(0)
        return {"text": source, "text_encoded": enc}

    def process_helper(self, save_path: str, **kwargs):
        train = load_dataset("tiny_shakespeare", split="train")
        val = load_dataset("tiny_shakespeare", split="validation")
        train = train.map(self.encode)
        val = val.map(self.encode)
        train_path = self.format_dataset_path(save_path, DataloaderType.TRAIN)
        val_path = self.format_dataset_path(save_path, DataloaderType.VALIDATION)
        train.save_to_disk(train_path)
        val.save_to_disk(val_path)

    def collate_causal_fn(self) -> Callable[[List[List[int]], int, int, int], Tensor]:
        def collate_fn(
            batch: List[List[int]], bos_idx: int, eos_idx: int, pad_idx: int
        ) -> Tensor:
            bos_idx = torch.tensor([bos_idx], dtype=torch.long)
            eos_idx = torch.tensor([eos_idx], dtype=torch.long)
            combined = [torch.cat([bos_idx, torch.tensor(x), eos_idx]) for x in batch]

            source = [x[:-1] for x in combined]
            target = [x[1:] for x in combined]

            source = torch.nested.nested_tensor(source)
            target = torch.nested.nested_tensor(target)

            source = torch.nested.to_padded_tensor(source, pad_idx)
            target = torch.nested.to_padded_tensor(target, pad_idx)

            return source, target

        return collate_fn

    def causal(
        self, dataset_path: str, type: DataloaderType, **kwargs
    ) -> DataLoader | None:
        path = self.format_dataset_path(dataset_path, type)
        batch_size = kwargs["batch_size"]
        max_length = kwargs["max_length"]

        class DataTransform(Dataset):
            def __init__(self, dataset_path: str, max_length: int):
                self.path = dataset_path
                self.max_length = max_length
                self.raw = load_from_disk(self.path)

            def __len__(self):
                return len(self.raw["text_encoded"][0])

            def __getitem__(self, index) -> Any:
                return self.raw["text_encoded"][0][index : index + self.max_length]

        dt = DataTransform(path, max_length)
        collate = self.collate_causal_fn()
        dl = DataLoader(
            dt,
            batch_size=batch_size,
            collate_fn=lambda x: collate(x, self.bos_idx, self.eos_idx, self.pad_idx),
        )

        return dl

    def causal_verify_args(self, **kwargs) -> Verification:
        return verify_args(
            {
                "batch_size": ArgInfo(
                    level=ParamLevel.REQUIRED,
                    type=int,
                    description="Batch size for the dataloader",
                ),
                "max_length": ArgInfo(
                    level=ParamLevel.REQUIRED,
                    type=int,
                    description="Max length of sequence used to train the model",
                ),
            },
            **kwargs
        )