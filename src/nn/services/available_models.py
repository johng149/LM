from src.nn.architectures.decoder import Decoder
from src.datasets.base.processor import Processor
from typing import Callable, Optional
import torch

available_models = {
    "decoder": Decoder,
}


def model_type_to_processor_dataloader(
    p: Processor, model_type: str
) -> Optional[Callable]:
    match model_type:
        case "decoder":
            return p.causal
        case _:
            return None


def model_type_to_processor_supports(
    p: Processor, model_type: str
) -> Optional[Callable]:
    match model_type:
        case "decoder":
            return p.supports_causal
        case _:
            return False


def model_type_to_processor_verify_args(
    p: Processor, model_type: str
) -> Optional[Callable]:
    match model_type:
        case "decoder":
            return p.causal_verify_args
        case _:
            return None
