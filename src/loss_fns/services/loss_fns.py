from src.loss_fns.services.cross_entropy import create_cross_entropy_loss_fn
from src.loss_fns.services.dag_loss import brute_force_dag_loss
from src.loss_fns.services.dag_loss_efficient import dag_loss_adapter

available_loss_fns = {
    "cross_entropy": create_cross_entropy_loss_fn,
    # below we use dummy lambda that takes in an `x`
    # (the tokenizer), so that it follows the same
    # pattern as the other entries in the dictionary
    "brute_force_dag_loss": lambda x: brute_force_dag_loss,
    "dag_loss": lambda x: dag_loss_adapter,
}
