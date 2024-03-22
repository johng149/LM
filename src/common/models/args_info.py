from typing import TypedDict, Any, List
from src.common.models.param_level import ParamLevel

ArgInfo = TypedDict(
    "ArgInfo",
    {
        "level": ParamLevel,
        "description": str,
        "type": Any
    }
)

ArgsInfo = TypedDict(
    "ArgsInfo",
    {
        str: ArgInfo
    }
)