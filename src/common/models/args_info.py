from typing import TypedDict, Any, List
from src.common.models.param_level import ParamLevel


class ArgInfo:
    def __init__(self, level: ParamLevel, description: str, type: Any):
        self.level = level
        self.description = description
        self.type = type

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ArgInfo):
            return False
        return (
            self.level == value.level
            and self.description == value.description
            and self.type == value.type
        )


ArgsInfo = TypedDict("ArgsInfo", {str: ArgInfo})
