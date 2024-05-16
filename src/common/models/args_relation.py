from typing import Callable, Tuple, TypedDict
from src.common.models.param_level import ParamLevel


class ArgRelation:
    def __init__(
        self, level: ParamLevel, relation: Callable[..., bool], failure_msg: str
    ):
        self.level = level
        self.relation = relation
        self.failure_msg = failure_msg

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ArgRelation):
            return False
        return (
            self.level == value.level
            and self.relation == value.relation
            and self.failure_msg == value.failure_msg
        )


# due to limitations with how typing works, please ensure that when using this,
# the callable accepts the same number of arguments as the tuple has elements
ArgRelations = TypedDict("ArgRelations", {Tuple[str, ...]: ArgRelation})
