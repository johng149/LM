from typing import TypedDict, Any, List
from src.common.models.param_level import ParamLevel


class ArgInfo:
    def __init__(self, level: ParamLevel, description: str, type: Any):
        self.level = level
        self.description = description
        self.type = type


ArgsInfo = TypedDict("ArgsInfo", {str: ArgInfo})
