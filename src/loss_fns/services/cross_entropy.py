from torch.nn import functional as F
from src.tokenizers.models.info import Info


def create_cross_entropy_loss_fn(info: Info):
    def cross_entropy(logits, labels):
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1)
        return F.cross_entropy(logits, labels, ignore_index=info.pad_idx)

    return cross_entropy
