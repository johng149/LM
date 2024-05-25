from torch.nn import functional as F
from src.tokenizers.models.info import Info
from src.loss_fns.services.cross_entropy import create_cross_entropy_loss_fn


available_loss_fns = {"cross_entropy": create_cross_entropy_loss_fn}
