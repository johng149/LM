from src.common.models.args_info import ArgsInfo, ArgInfo
from src.common.models.param_level import ParamLevel
from src.common.models.verification import Verification, Error, Warning
from typing import List, Tuple


def unknown_arg_msg(arg: str) -> str:
    return f"Unknown argument: {arg}"


def missing_arg_msg(arg: str) -> str:
    return f"Missing required argument: {arg}"


def type_mismatch_msg(arg: str, expected_type: type) -> str:
    return f"Argument {arg} should be of type {expected_type}"


def missing_suggested_arg_msg(arg: str) -> str:
    return f"Missing suggested argument: {arg}"


def verify_args(info: ArgsInfo, **kwargs) -> Tuple[List[Verification], bool]:
    """
    Verify the arguments passed to the function.

    Will produce warning if kwargs include keywords that are not
    in the given info.
    """
    result = []
    hasError = False
    known_args = set(info.keys())
    given_args = set(kwargs.keys())
    unknown_args = given_args - known_args
    for arg in unknown_args:
        result.append(Warning(unknown_arg_msg(arg)))
    for item in info.items():
        arg: str = item[0]
        arg_info: ArgInfo = item[1]
        level = arg_info.level
        desc = arg_info.description
        expected_type = arg_info.type
        match level:
            case ParamLevel.REQUIRED:
                if arg not in kwargs:
                    result.append(Error(missing_arg_msg(arg)))
                    hasError = True
                elif not isinstance(kwargs[arg], expected_type):
                    result.append(Error(type_mismatch_msg(arg, expected_type)))
                    hasError = True
            case ParamLevel.SUGGESTED:
                if arg not in kwargs:
                    result.append(Warning(missing_suggested_arg_msg(arg)))
                elif not isinstance(kwargs[arg], expected_type):
                    result.append(Error(type_mismatch_msg(arg, expected_type)))
                    hasError = True
            case ParamLevel.OPTIONAL:
                if arg in kwargs and not isinstance(kwargs[arg], expected_type):
                    result.append(Error(type_mismatch_msg(arg, expected_type)))
                    hasError = True
    return result, hasError
