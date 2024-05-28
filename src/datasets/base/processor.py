from src.tokenizers.models.info import Info
from src.common.models.verification import Verification
from src.common.services.verification import (
    verify_args,
    verify_arg_relations,
    combine_verification_results,
)
from src.common.models.dataloader_type import DataloaderType
from torch.utils.data import DataLoader
from typing import Optional, Callable, Any
import json
from pathlib import Path
from torch import Tensor
from typing import List, Tuple, Optional


class Processor:

    def __init__(self, info: Info):
        self.info = info
        self.tokenizer = info.tokenizer
        self.tokenizer_name = info.tokenizer_name
        self.pad_idx = info.pad_idx
        self.bos_idx = info.bos_idx
        self.eos_idx = info.eos_idx
        self.mask_idx = info.mask_idx
        self.vocab_size = info.vocab_size

    def process_helper(self, save_path: str, format: Optional[str], **kwargs):
        raise NotImplementedError

    def process(self, save_path: str, use_cache=True, **kwargs):
        already_cached = self.already_cached(
            save_path, self.__class__.__name__, **kwargs
        )
        if use_cache and already_cached:
            return
        self.process_helper(save_path, **kwargs)
        self.set_cache(save_path, self.__class__.__name__, **kwargs)

    def process_verify_args_helper(
        self, **kwargs
    ) -> Tuple[List[Verification], bool, List[Verification], bool]:
        v1, e1 = verify_args({}, **kwargs)
        v2, e2 = verify_arg_relations({}, **kwargs)
        return v1, e1, v2, e2

    def process_verify_args(self, **kwargs) -> Tuple[List[Verification], bool]:
        v1, e1, v2, e2 = self.process_verify_args_helper(**kwargs)
        return combine_verification_results([v1, v2, e1, e2])

    def already_cached(self, path: str, class_name: str, **kwargs) -> bool:
        # we look for a json file at the specified path, we expect
        # that the file is named after the class_name and that it
        # contains the kwargs as a dictionary
        expected_path = self.format_cache_path(path, class_name)
        if not expected_path.exists():
            return False
        with open(expected_path, "r") as f:
            expected_kwargs = json.load(f)
        return expected_kwargs == self.format_kwargs(**kwargs)

    def set_cache(self, path: str, class_name: str, **kwargs):
        # we save the kwargs as a json file at the specified path
        expected_path = self.format_cache_path(path, class_name)
        with open(expected_path, "w") as f:
            json.dump(self.format_kwargs(**kwargs), f)

    def format_kwargs(self, **kwargs) -> dict:
        """
        Given a set of kwargs, it returns a dictionary with the
        given kwargs as well as information about the tokenizer
        that was used to initialize the processor.

        @param kwargs: the kwargs to be formatted
        @return: the formatted kwargs
        """
        return {
            "kwargs": kwargs,
            "tokenizer_info": {
                "tokenizer_name": self.tokenizer_name,
                "pad_idx": self.pad_idx,
                "bos_idx": self.bos_idx,
                "eos_idx": self.eos_idx,
                "mask_idx": self.mask_idx,
                "vocab_size": self.vocab_size,
            },
        }

    def encode(self, sample):
        raise NotImplementedError

    def seq2seq(
        self, dataset_path: str, type: DataloaderType, batch_size: int, **kwargs
    ) -> Optional[DataLoader]:
        return None

    @staticmethod
    def naive_inference_seq2seq(
        bos_idx: int | Tensor, eos_idx: int | Tensor, *args: Tensor
    ):
        """
        Given the encoder and decoder inputs (which are produced by
        tokenizing the respective input text), this function returns
        the encoder and decoder inputs that should be fed to the model
        for naive inference. It should reflect how the sequences are
        transformed by the seq2seq collate function

        If bos_idx and eos_idx are tensors, they should be of shape
        [1], that is, equivalent to torch.tensor([bos_idx], dtype=torch.long)
        etc

        It uses *args instead of specifying two inputs (one for the encoder
        and one for the decoder) since we don't know the exact way children
        classes might implement the collate function
        """
        raise NotImplementedError

    def seq2seq_verify_args(self, **kwargs) -> Tuple[List[Verification], bool]:
        return verify_args({}, **kwargs)

    def collate_seq2seq_fn(
        self,
    ) -> Optional[
        Callable[
            [Any, int, int, int], Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        ]
    ]:
        """
        For datasets that support seq2seq language modeling, this function
        should return a collate function that given batch some data which
        contains the batch, for example it could be a list of list of integers
        or a list of tensors, or some other format
        bos_idx (the index of the beginning of sequence token),
        eos_idx (the index of the end of sequence token),
        pad_idx (the index of the padding token),
        returns a tuple of tensors
        (encoder_input,
        encoder_self_attn_mask,
        decoder_input,
        decoder_self_attn_mask,
        encoder_kv_decoder_q_mask,
        target)
        """
        return None

    def supports_seq2seq(self) -> bool:
        return self.collate_seq2seq_fn() is not None

    def causal(
        self, dataset_path: str, type: DataloaderType, batch_size: int, **kwargs
    ) -> Optional[DataLoader]:
        return None

    @staticmethod
    def naive_inference_causal(
        bos_idx: int | Tensor, eos_idx: int | Tensor, *args: Tensor
    ):
        """
        Given the decoder input (which is produced by tokenizing the
        input text), this function returns the decoder input that should
        be fed to the model for naive inference. It should reflect how
        the sequences are transformed by the causal collate function

        If bos_idx and eos_idx are tensors, they should be of shape
        [1], that is, equivalent to torch.tensor([bos_idx], dtype=torch.long)
        etc

        It uses *args instead of specifying one input since we don't know
        the exact way children classes might implement the collate function
        """
        raise NotImplementedError

    def causal_verify_args(self, **kwargs) -> Tuple[List[Verification], bool]:
        return verify_args({}, **kwargs)

    def collate_causal_fn(
        self,
    ) -> Optional[Callable[[Any, int, int, int], Tuple[Tensor, Tensor, Tensor]]]:
        """
        For datasets that support causal language modeling, this
        function should return a collate function that given
        batch some data which contains the batch, for example,
            it could be a list of list of integers or a list of
            tensors, or some other format
        bos_idx (the index of the beginning of sequence token),
        eos_idx (the index of the end of sequence token),
        pad_idx (the index of the padding token),
        return a tuple of tensors (source, target) where source
        is the input to the model and target is the expected
        output of the model.
        """
        return None

    def supports_causal(self) -> bool:
        return self.collate_causal_fn() is not None

    def seq2seq_dag(
        self, dataset_path: str, type: DataloaderType, batch_size: int, **kwargs
    ) -> Optional[DataLoader]:
        return None

    @staticmethod
    def naive_inference_seq2seq_dag(
        bos_idx: int | Tensor, eos_idx: int | Tensor, *args: Tensor
    ):
        raise NotImplementedError

    def seq2seq_dag_verify_args(self, **kwargs) -> Tuple[List[Verification], bool]:
        return verify_args({}, **kwargs)

    def collate_seq2seq_dag_fn(self, coeff_fn: Callable[[], int]) -> Optional[
        Callable[
            [Any, int, int, int],
            Tuple[
                Tensor,
                Tensor,
                Tensor,
                Tensor,
                Tensor,
                Tensor,
                Tuple[Tensor, Tensor, Tensor],
            ],
        ]
    ]:
        """
        The DAG dataset is a bit different from the regular seq2seq dataset
        We take in the data batch, the bos_idx, eos_idx, and pad_idx
        and return a tuple of tensors. The decoder input for the model
        depends on the size of the encoder input and the output of the
        coeff_fn function.

        The output is:
        encoder_input,
        enc_pad_mask,
        decoder_input,
        dec_pad_mask,
        env_kv_dec_q_mask,
        vertex_lens,
        target_lens,
        targets

        where vertex_lens, target_lens, and targets are used to calculate
        the loss for the model ( and since the codebase was built with the
        assumption that only the final element is used to calculate loss
        we need to put these 3 in a tuple together)
        """
        return None

    def supports_seq2seq_dag(self) -> bool:
        # we use a dummy coeff_fn since this is just to check if the
        # collate_seq2seq_dag_fn function is implemented
        return self.collate_seq2seq_dag_fn(lambda: 1) is not None

    def format_dataset_path(self, dataset_path: str, type: DataloaderType) -> Path:
        dataset_path = Path(dataset_path)
        match type:
            case DataloaderType.TRAIN:
                return dataset_path / "train"
            case DataloaderType.VALIDATION:
                return dataset_path / "val"
            case _:
                raise ValueError(f"Invalid type: {type}")

    def format_cache_path(self, cache_path: str, class_name: str) -> Path:
        return Path(cache_path) / f"{class_name}.json"
