from src.loss_fns.services.cross_entropy import create_cross_entropy_loss_fn
from src.loss_fns.services.dag_loss import brute_force_dag_loss
from src.loss_fns.services.dag_loss_efficient import dag_loss_adapter

available_loss_fns = {
    "cross_entropy": create_cross_entropy_loss_fn,
    "brute_force_dag_loss": lambda: brute_force_dag_loss,
    "dag_loss": lambda: dag_loss_adapter,
}
