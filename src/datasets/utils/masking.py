import torch
from torch import Tensor

def causal_mask(input_tensor: Tensor, pad_idx: int) -> Tensor:
    """
    Create causal mask for the input tensor. The mask
    is a lower triangular matrix with the diagonal and
    has the appropriate columns zeroed out for positions
    that are padding tokens

    @param input_tensor: the input tensor of shape 
        (batch_size, seq_len)
    @return: causal mask of shape (batch_size, seq_len, seq_len)
    """
    batch_size, seq_len = input_tensor.shape
    token_is_not_pad = (input_tensor != pad_idx)
    padding_mask = token_is_not_pad.unsqueeze(-1).expand(-1, -1, seq_len).transpose(-2, -1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_tensor.device)).unsqueeze(0).expand(batch_size, -1, -1)
    return padding_mask & causal_mask.bool()