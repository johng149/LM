from src.loss_fns.services.cross_entropy import create_cross_entropy_loss_fn
from src.loss_fns.services.dag_loss import brute_force_dag_loss

available_loss_fns = {
    "cross_entropy": create_cross_entropy_loss_fn,
    "brute_force_dag_loss": lambda: brute_force_dag_loss,
}
