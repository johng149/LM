import random
from typing import Callable


def random_coeff_fn_maker(high: int) -> Callable[[], int]:
    def random_coeff_fn() -> int:
        return random.randint(1, high)

    return random_coeff_fn


available_coeff_fns = {
    "random": random_coeff_fn_maker,
}
