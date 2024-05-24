from torch.nn import Module
from src.nn.models.decoding_strat_model import AutoregressiveStrategy
from torch import Tensor
from src.common.models.verification import Verification
from typing import List, Tuple
from src.common.services.verification import (
    verify_args,
    verify_arg_relations,
    combine_verification_results,
)


class Architecture(Module):
    """
    This is an abstract class that represents a neural network architecture.
    """

    def init_kwargs(self) -> dict:
        """
        Returns the kwargs used to initialize the architecture.
        """
        raise NotImplementedError

    def verify_init_kwargs(self, **kwargs) -> Tuple[List[Verification], bool]:
        v1, e1, v2, e2 = self.verify_init_kwargs_helper(**kwargs)
        return combine_verification_results([v1, v2, e1, e2])

    def verify_init_kwargs_helper(
        self, **kwargs
    ) -> Tuple[List[Verification], bool, List[Verification], bool]:
        v1, e1 = verify_args({}, **kwargs)
        v2, e2 = verify_arg_relations({}, **kwargs)
        return v1, e1, v2, e2

    def naive_inference(
        self, *args, strat: AutoregressiveStrategy, max_len: int
    ) -> Tensor:
        """
        Runs inference assuming that `x` has batch size 1, and contains
        no padding tokens.

        @param x: the input tensor
        @param strat: the decoding strategy to use
        @param max_len: the maximum length of the output tensor
        @return: the output tensor
        """
        return None
