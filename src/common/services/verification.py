from src.common.models.args_info import ArgsInfo, ArgInfo
from src.common.models.args_relation import ArgRelations, ArgRelation
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


def args_tuple_in_kwargs(args: Tuple[str], kwargs: dict) -> Tuple[bool, List[str]]:
    """
    Check if all the arguments in the tuple are in the kwargs.
    """
    missing_args = []
    for arg in args:
        if arg not in kwargs:
            missing_args.append(arg)
    return len(missing_args) != 0, missing_args


def args_missing_msg(args: List[str]) -> str:
    return f"Missing arguments: {args}"


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
                elif not type(kwargs[arg]) == expected_type:
                    result.append(Error(type_mismatch_msg(arg, expected_type)))
                    hasError = True
            case ParamLevel.SUGGESTED:
                # similar to optional, but we will warn if the argument is missing
                if arg not in kwargs:
                    result.append(Warning(missing_suggested_arg_msg(arg)))
                elif not type(kwargs[arg]) == expected_type:
                    result.append(Error(type_mismatch_msg(arg, expected_type)))
                    hasError = True
            case ParamLevel.OPTIONAL:
                # optional means that if the argument is present, it should be of the correct type
                # but if it is not present, that is fine
                if arg in kwargs and not type(kwargs[arg]) == expected_type:
                    result.append(Error(type_mismatch_msg(arg, expected_type)))
                    hasError = True
    return result, hasError


def verify_arg_relations(
    relations: ArgRelations, **kwargs
) -> Tuple[List[Verification], bool]:
    """
    Verify the relationships between arguments passed to the function.
    """
    result = []
    hasError = False
    for item in relations.items():
        args: Tuple[str, ...] = item[0]
        relation: ArgRelation = item[1]
        level = relation.level
        failure_msg = relation.failure_msg
        relation_fn = relation.relation
        missing, missing_args = args_tuple_in_kwargs(args, kwargs)
        match level:
            case ParamLevel.REQUIRED:
                if missing:
                    result.append(Error(args_missing_msg(missing_args)))
                    hasError = True
                elif not relation_fn(*[kwargs[arg] for arg in args]):
                    result.append(Error(failure_msg))
                    hasError = True
            case ParamLevel.SUGGESTED:
                # similar to optional, but we will warn if the argument(s) are missing
                if missing:
                    result.append(Warning(args_missing_msg(missing_args)))
                elif not relation_fn(*[kwargs[arg] for arg in args]):
                    result.append(Error(failure_msg))
                    hasError = True
            case ParamLevel.OPTIONAL:
                # optional means that if the argument(s) are present, they should satisfy the relation
                # but if they are not present, that is fine
                if not missing and not relation_fn(*[kwargs[arg] for arg in args]):
                    result.append(Error(failure_msg))
                    hasError = True
    return result, hasError
