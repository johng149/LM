from src.nn.architectures.decoder import Decoder
from src.nn.architectures.encoder_decoder import EncoderDecoder
from src.datasets.base.processor import Processor
from typing import Callable, Optional
import torch

available_models = {
    "decoder": Decoder,
    "seq2seq": EncoderDecoder,
}


def model_type_to_processor_dataloader(
    p: Processor, model_type: str
) -> Optional[Callable]:
    match model_type:
        case "decoder":
            return p.causal
        case "seq2seq":
            return p.seq2seq
        case _:
            return None


def model_type_to_processor_supports(
    p: Processor, model_type: str
) -> Optional[Callable]:
    match model_type:
        case "decoder":
            return p.supports_causal
        case "seq2seq":
            return p.supports_seq2seq
        case _:
            return False


def model_type_to_processor_verify_args(
    p: Processor, model_type: str
) -> Optional[Callable]:
    match model_type:
        case "decoder":
            return p.causal_verify_args
        case "seq2seq":
            return p.seq2seq_verify_args
        case _:
            return None
