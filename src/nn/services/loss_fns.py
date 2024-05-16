from torch.nn import functional as F


def cross_entropy(logits, labels):
    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.reshape(-1, vocab_size)
    labels = labels.reshape(-1)
    return F.cross_entropy(logits, labels)


available_loss_fns = {"cross_entropy": cross_entropy}
