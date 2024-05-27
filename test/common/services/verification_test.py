from src.common.services.verification import verify_args
from src.common.services.verification import verify_arg_relations
from src.common.services.verification import unknown_arg_msg as unk
from src.common.services.verification import missing_arg_msg as miss
from src.common.services.verification import type_mismatch_msg as typmm
from src.common.services.verification import missing_suggested_arg_msg as miss_sugg
from src.common.services.verification import args_missing_msg as miss_arg
from src.common.models.verification import Warning, Error
from src.common.models.param_level import ParamLevel
from src.common.models.args_info import ArgInfo
from src.common.models.args_relation import ArgRelation
from unittest.mock import patch
from unittest import mock
from typing import Callable

# prior reading:
# https://engineeringblog.yelp.com/2015/02/assert_called_once-threat-or-menace.html


@patch("src.common.services.verification.unknown_arg_msg", wraps=unk)
@patch("src.common.services.verification.missing_arg_msg", wraps=miss)
@patch("src.common.services.verification.type_mismatch_msg", wraps=typmm)
@patch("src.common.services.verification.missing_suggested_arg_msg", wraps=miss_sugg)
def test_verify_args_empty_info(mock_miss_sugg, mock_typmm, mock_miss, mock_unk):
    info = {}
    kwargs = {}
    result, hasError = verify_args(info, **kwargs)
    assert not hasError
    assert len(result) == 0
    assert mock_miss_sugg.call_count == 0
    assert mock_typmm.call_count == 0
    assert mock_miss.call_count == 0
    assert mock_unk.call_count == 0


@patch("src.common.services.verification.unknown_arg_msg", wraps=unk)
@patch("src.common.services.verification.missing_arg_msg", wraps=miss)
@patch("src.common.services.verification.type_mismatch_msg", wraps=typmm)
@patch("src.common.services.verification.missing_suggested_arg_msg", wraps=miss_sugg)
def test_verify_args_unknown_arg(mock_miss_sugg, mock_typmm, mock_miss, mock_unk):
    info = {}
    arg_name = "some_unknown_arg"
    arg_value = "some_value"
    kwargs = {arg_name: arg_value}
    result, hasError = verify_args(info, **kwargs)
    assert not hasError
    assert len(result) == 1
    assert isinstance(result[0], Warning)
    assert result[0].message == unk(arg_name)
    assert mock_miss_sugg.call_count == 0
    assert mock_typmm.call_count == 0
    assert mock_miss.call_count == 0
    assert mock_unk.call_count == 1
    assert mock_unk.call_args == mock.call(arg_name)


@patch("src.common.services.verification.unknown_arg_msg", wraps=unk)
@patch("src.common.services.verification.missing_arg_msg", wraps=miss)
@patch("src.common.services.verification.type_mismatch_msg", wraps=typmm)
@patch("src.common.services.verification.missing_suggested_arg_msg", wraps=miss_sugg)
def test_verify_args_type_mismatch_arg_optional(
    mock_miss_sugg, mock_typmm, mock_miss, mock_unk
):
    arg_name = "some_arg"
    arg_type = int
    info = {
        arg_name: ArgInfo(
            level=ParamLevel.OPTIONAL, description="some description", type=arg_type
        )
    }
    kwargs = {arg_name: "some_value"}
    result, hasError = verify_args(info, **kwargs)
    assert hasError
    assert len(result) == 1
    assert isinstance(result[0], Error)
    assert result[0].message == typmm(arg_name, arg_type)
    assert mock_miss_sugg.call_count == 0
    assert mock_typmm.call_count == 1
    assert mock_miss.call_count == 0
    assert mock_unk.call_count == 0
    assert mock_typmm.call_args == mock.call(arg_name, arg_type)


@patch("src.common.services.verification.unknown_arg_msg", wraps=unk)
@patch("src.common.services.verification.missing_arg_msg", wraps=miss)
@patch("src.common.services.verification.type_mismatch_msg", wraps=typmm)
@patch("src.common.services.verification.missing_suggested_arg_msg", wraps=miss_sugg)
def test_verify_args_type_mismatch_arg_optional_callable(
    mock_miss_sugg, mock_typmm, mock_miss, mock_unk
):
    arg_name = "some_arg"
    arg_type = Callable
    info = {
        arg_name: ArgInfo(
            level=ParamLevel.OPTIONAL, description="some description", type=arg_type
        )
    }
    kwargs = {arg_name: "some_value"}
    result, hasError = verify_args(info, **kwargs)
    assert hasError
    assert len(result) == 1
    assert isinstance(result[0], Error)
    assert result[0].message == typmm(arg_name, arg_type)
    assert mock_miss_sugg.call_count == 0
    assert mock_typmm.call_count == 1
    assert mock_miss.call_count == 0
    assert mock_unk.call_count == 0
    assert mock_typmm.call_args == mock.call(arg_name, arg_type)


@patch("src.common.services.verification.unknown_arg_msg", wraps=unk)
@patch("src.common.services.verification.missing_arg_msg", wraps=miss)
@patch("src.common.services.verification.type_mismatch_msg", wraps=typmm)
@patch("src.common.services.verification.missing_suggested_arg_msg", wraps=miss_sugg)
def test_verify_args_type_mismatch_arg_suggested(
    mock_miss_sugg, mock_typmm, mock_miss, mock_unk
):
    arg_name = "some_arg"
    arg_type = int
    info = {
        arg_name: ArgInfo(
            level=ParamLevel.SUGGESTED, description="some description", type=arg_type
        )
    }
    kwargs = {arg_name: "some_value"}
    result, hasError = verify_args(info, **kwargs)
    assert hasError
    assert len(result) == 1
    assert isinstance(result[0], Error)
    assert result[0].message == typmm(arg_name, arg_type)
    assert mock_miss_sugg.call_count == 0
    assert mock_typmm.call_count == 1
    assert mock_miss.call_count == 0
    assert mock_unk.call_count == 0
    assert mock_typmm.call_args == mock.call(arg_name, arg_type)


@patch("src.common.services.verification.unknown_arg_msg", wraps=unk)
@patch("src.common.services.verification.missing_arg_msg", wraps=miss)
@patch("src.common.services.verification.type_mismatch_msg", wraps=typmm)
@patch("src.common.services.verification.missing_suggested_arg_msg", wraps=miss_sugg)
def test_verify_args_type_mismatch_arg_suggested_callable(
    mock_miss_sugg, mock_typmm, mock_miss, mock_unk
):
    arg_name = "some_arg"
    arg_type = Callable
    info = {
        arg_name: ArgInfo(
            level=ParamLevel.SUGGESTED, description="some description", type=arg_type
        )
    }
    kwargs = {arg_name: "some_value"}
    result, hasError = verify_args(info, **kwargs)
    assert hasError
    assert len(result) == 1
    assert isinstance(result[0], Error)
    assert result[0].message == typmm(arg_name, arg_type)
    assert mock_miss_sugg.call_count == 0
    assert mock_typmm.call_count == 1
    assert mock_miss.call_count == 0
    assert mock_unk.call_count == 0
    assert mock_typmm.call_args == mock.call(arg_name, arg_type)


@patch("src.common.services.verification.unknown_arg_msg", wraps=unk)
@patch("src.common.services.verification.missing_arg_msg", wraps=miss)
@patch("src.common.services.verification.type_mismatch_msg", wraps=typmm)
@patch("src.common.services.verification.missing_suggested_arg_msg", wraps=miss_sugg)
def test_verify_args_type_mismatch_arg_required(
    mock_miss_sugg, mock_typmm, mock_miss, mock_unk
):
    arg_name = "some_arg"
    arg_type = int
    info = {
        arg_name: ArgInfo(
            level=ParamLevel.REQUIRED, description="some description", type=arg_type
        )
    }
    kwargs = {arg_name: "some_value"}
    result, hasError = verify_args(info, **kwargs)
    assert hasError
    assert len(result) == 1
    assert isinstance(result[0], Error)
    assert result[0].message == typmm(arg_name, arg_type)
    assert mock_miss_sugg.call_count == 0
    assert mock_typmm.call_count == 1
    assert mock_miss.call_count == 0
    assert mock_unk.call_count == 0
    assert mock_typmm.call_args == mock.call(arg_name, arg_type)


@patch("src.common.services.verification.unknown_arg_msg", wraps=unk)
@patch("src.common.services.verification.missing_arg_msg", wraps=miss)
@patch("src.common.services.verification.type_mismatch_msg", wraps=typmm)
@patch("src.common.services.verification.missing_suggested_arg_msg", wraps=miss_sugg)
def test_verify_args_type_mismatch_arg_required_callable(
    mock_miss_sugg, mock_typmm, mock_miss, mock_unk
):
    arg_name = "some_arg"
    arg_type = Callable
    info = {
        arg_name: ArgInfo(
            level=ParamLevel.REQUIRED, description="some description", type=arg_type
        )
    }
    kwargs = {arg_name: "some_value"}
    result, hasError = verify_args(info, **kwargs)
    assert hasError
    assert len(result) == 1
    assert isinstance(result[0], Error)
    assert result[0].message == typmm(arg_name, arg_type)
    assert mock_miss_sugg.call_count == 0
    assert mock_typmm.call_count == 1
    assert mock_miss.call_count == 0
    assert mock_unk.call_count == 0
    assert mock_typmm.call_args == mock.call(arg_name, arg_type)


@patch("src.common.services.verification.unknown_arg_msg", wraps=unk)
@patch("src.common.services.verification.missing_arg_msg", wraps=miss)
@patch("src.common.services.verification.type_mismatch_msg", wraps=typmm)
@patch("src.common.services.verification.missing_suggested_arg_msg", wraps=miss_sugg)
def test_verify_args_missing_suggested_arg(
    mock_miss_sugg, mock_typmm, mock_miss, mock_unk
):
    arg_name = "some_arg"
    arg_type = int
    info = {
        arg_name: ArgInfo(
            level=ParamLevel.SUGGESTED, description="some description", type=arg_type
        )
    }
    kwargs = {}
    result, hasError = verify_args(info, **kwargs)
    assert not hasError
    assert len(result) == 1
    assert isinstance(result[0], Warning)
    assert result[0].message == miss_sugg(arg_name)
    assert mock_miss_sugg.call_count == 1
    assert mock_typmm.call_count == 0
    assert mock_miss.call_count == 0
    assert mock_unk.call_count == 0
    assert mock_miss_sugg.call_args == mock.call(arg_name)


@patch("src.common.services.verification.unknown_arg_msg", wraps=unk)
@patch("src.common.services.verification.missing_arg_msg", wraps=miss)
@patch("src.common.services.verification.type_mismatch_msg", wraps=typmm)
@patch("src.common.services.verification.missing_suggested_arg_msg", wraps=miss_sugg)
def test_verify_args_missing_required_arg(
    mock_miss_sugg, mock_typmm, mock_miss, mock_unk
):
    arg_name = "some_arg"
    arg_type = int
    info = {
        arg_name: ArgInfo(
            level=ParamLevel.REQUIRED, description="some description", type=arg_type
        )
    }
    kwargs = {}
    result, hasError = verify_args(info, **kwargs)
    assert hasError
    assert len(result) == 1
    assert isinstance(result[0], Error)
    assert result[0].message == miss(arg_name)
    assert mock_miss_sugg.call_count == 0
    assert mock_typmm.call_count == 0
    assert mock_miss.call_count == 1
    assert mock_unk.call_count == 0
    assert mock_miss.call_args == mock.call(arg_name)


@patch("src.common.services.verification.unknown_arg_msg", wraps=unk)
@patch("src.common.services.verification.missing_arg_msg", wraps=miss)
@patch("src.common.services.verification.type_mismatch_msg", wraps=typmm)
@patch("src.common.services.verification.missing_suggested_arg_msg", wraps=miss_sugg)
def test_verify_args_missing_optional_arg(
    mock_miss_sugg, mock_typmm, mock_miss, mock_unk
):
    arg_name = "some_arg"
    arg_type = int
    info = {
        arg_name: ArgInfo(
            level=ParamLevel.OPTIONAL, description="some description", type=arg_type
        )
    }
    kwargs = {}
    result, hasError = verify_args(info, **kwargs)
    assert not hasError
    assert len(result) == 0
    assert mock_miss_sugg.call_count == 0
    assert mock_typmm.call_count == 0
    assert mock_miss.call_count == 0
    assert mock_unk.call_count == 0


@patch("src.common.services.verification.unknown_arg_msg", wraps=unk)
@patch("src.common.services.verification.missing_arg_msg", wraps=miss)
@patch("src.common.services.verification.type_mismatch_msg", wraps=typmm)
@patch("src.common.services.verification.missing_suggested_arg_msg", wraps=miss_sugg)
def test_verify_args_successful(mock_miss_sugg, mock_typmm, mock_miss, mock_unk):
    arg_optional_name = "some_optional_arg"
    arg_optional_type = int
    arg_required_name = "some_required_arg"
    arg_required_type = str
    arg_suggested_name = "some_suggested_arg"
    arg_suggested_type = float
    info = {
        arg_optional_name: ArgInfo(
            level=ParamLevel.OPTIONAL,
            description="some description",
            type=arg_optional_type,
        ),
        arg_required_name: ArgInfo(
            level=ParamLevel.REQUIRED,
            description="some description",
            type=arg_required_type,
        ),
        arg_suggested_name: ArgInfo(
            level=ParamLevel.SUGGESTED,
            description="some description",
            type=arg_suggested_type,
        ),
    }
    kwargs = {
        arg_optional_name: 1,
        arg_required_name: "some_value",
        arg_suggested_name: 1.0,
    }
    result, hasError = verify_args(info, **kwargs)
    assert not hasError
    assert len(result) == 0
    assert mock_miss_sugg.call_count == 0
    assert mock_typmm.call_count == 0
    assert mock_miss.call_count == 0
    assert mock_unk.call_count == 0


@patch("src.common.services.verification.unknown_arg_msg", wraps=unk)
@patch("src.common.services.verification.missing_arg_msg", wraps=miss)
@patch("src.common.services.verification.type_mismatch_msg", wraps=typmm)
@patch("src.common.services.verification.missing_suggested_arg_msg", wraps=miss_sugg)
def test_verify_args_successful_callable(
    mock_miss_sugg, mock_typmm, mock_miss, mock_unk
):
    arg_optional_name = "some_optional_arg"
    arg_optional_type = Callable
    arg_required_name = "some_required_arg"
    arg_required_type = Callable
    arg_suggested_name = "some_suggested_arg"
    arg_suggested_type = Callable
    info = {
        arg_optional_name: ArgInfo(
            level=ParamLevel.OPTIONAL,
            description="some description",
            type=arg_optional_type,
        ),
        arg_required_name: ArgInfo(
            level=ParamLevel.REQUIRED,
            description="some description",
            type=arg_required_type,
        ),
        arg_suggested_name: ArgInfo(
            level=ParamLevel.SUGGESTED,
            description="some description",
            type=arg_suggested_type,
        ),
    }

    def suggested_example():
        return 3

    kwargs = {
        arg_optional_name: lambda x: x,
        arg_required_name: lambda x: x,
        arg_suggested_name: suggested_example,
    }
    result, hasError = verify_args(info, **kwargs)
    assert not hasError
    assert len(result) == 0
    assert mock_miss_sugg.call_count == 0
    assert mock_typmm.call_count == 0
    assert mock_miss.call_count == 0
    assert mock_unk.call_count == 0


@patch("src.common.services.verification.unknown_arg_msg", wraps=unk)
@patch("src.common.services.verification.missing_arg_msg", wraps=miss)
@patch("src.common.services.verification.type_mismatch_msg", wraps=typmm)
@patch("src.common.services.verification.missing_suggested_arg_msg", wraps=miss_sugg)
def test_verify_args_type_mismatch_arg_required_subtypes_strict(
    mock_miss_sugg, mock_typmm, mock_miss, mock_unk
):
    arg_name = "some_arg"
    arg_type = int
    info = {
        arg_name: ArgInfo(
            level=ParamLevel.REQUIRED, description="some description", type=arg_type
        )
    }
    kwargs = {arg_name: True}
    result, hasError = verify_args(info, **kwargs)
    assert hasError
    assert len(result) == 1
    assert isinstance(result[0], Error)
    assert result[0].message == typmm(arg_name, arg_type)
    assert mock_miss_sugg.call_count == 0
    assert mock_typmm.call_count == 1
    assert mock_miss.call_count == 0
    assert mock_unk.call_count == 0
    assert mock_typmm.call_args == mock.call(arg_name, arg_type)


def test_verify_arg_relations_empty_relations():
    relations = {}
    kwargs = {}
    result, hasError = verify_arg_relations(relations, **kwargs)
    assert not hasError
    assert len(result) == 0


def test_verify_arg_relations_required():
    arg1 = "arg1"
    arg2 = "arg2"
    relations = {
        (arg1, arg2): ArgRelation(
            level=ParamLevel.REQUIRED,
            failure_msg="some failure message",
            relation=lambda x, y: x == y,
        )
    }
    kwargs = {arg1: 1, arg2: 1}
    result, hasError = verify_arg_relations(relations, **kwargs)
    assert len(result) == 0
    assert not hasError


def test_verify_arg_relations_required_fail():
    arg1 = "arg1"
    arg2 = "arg2"
    relation = ArgRelation(
        level=ParamLevel.REQUIRED,
        failure_msg="some failure message",
        relation=lambda x, y: x == y,
    )
    relations = {(arg1, arg2): relation}
    kwargs = {arg1: 1, arg2: 2}
    result, hasError = verify_arg_relations(relations, **kwargs)
    assert len(result) == 1
    assert isinstance(result[0], Error)
    assert hasError
    assert result[0].message == relation.failure_msg


@patch("src.common.services.verification.args_missing_msg", wraps=miss_arg)
def test_verify_arg_relations_required_missing(mock_args_missing_msg):
    arg1 = "arg1"
    arg2 = "arg2"
    relation = ArgRelation(
        level=ParamLevel.REQUIRED,
        failure_msg="some failure message",
        relation=lambda x, y: x == y,
    )
    relations = {(arg1, arg2): relation}
    kwargs = {arg1: 1}
    result, hasError = verify_arg_relations(relations, **kwargs)
    assert len(result) == 1
    assert isinstance(result[0], Error)
    assert hasError
    assert result[0].message == miss_arg([arg2])
    assert mock_args_missing_msg.call_count == 1
    assert mock_args_missing_msg.call_args == mock.call([arg2])


def test_verify_arg_relations_suggested():
    arg1 = "arg1"
    arg2 = "arg2"
    relations = {
        (arg1, arg2): ArgRelation(
            level=ParamLevel.SUGGESTED,
            failure_msg="some failure message",
            relation=lambda x, y: x == y,
        )
    }
    kwargs = {arg1: 1, arg2: 1}
    result, hasError = verify_arg_relations(relations, **kwargs)
    assert len(result) == 0
    assert not hasError


def test_verify_arg_relations_suggested_fail():
    arg1 = "arg1"
    arg2 = "arg2"
    relation = ArgRelation(
        level=ParamLevel.SUGGESTED,
        failure_msg="some failure message",
        relation=lambda x, y: x == y,
    )
    relations = {(arg1, arg2): relation}
    kwargs = {arg1: 1, arg2: 2}
    result, hasError = verify_arg_relations(relations, **kwargs)
    assert len(result) == 1
    assert isinstance(result[0], Error)
    assert hasError
    assert result[0].message == relation.failure_msg


@patch("src.common.services.verification.args_missing_msg", wraps=miss_arg)
def test_verify_arg_relations_suggested_missing(mock_args_missing_msg):
    arg1 = "arg1"
    arg2 = "arg2"
    relation = ArgRelation(
        level=ParamLevel.SUGGESTED,
        failure_msg="some failure message",
        relation=lambda x, y: x == y,
    )
    relations = {(arg1, arg2): relation}
    kwargs = {arg1: 1}
    result, hasError = verify_arg_relations(relations, **kwargs)
    assert len(result) == 1
    assert isinstance(result[0], Warning)
    assert not hasError
    assert result[0].message == miss_arg([arg2])
    assert mock_args_missing_msg.call_count == 1
    assert mock_args_missing_msg.call_args == mock.call([arg2])


def test_verify_arg_relations_optional():
    arg1 = "arg1"
    arg2 = "arg2"
    relations = {
        (arg1, arg2): ArgRelation(
            level=ParamLevel.OPTIONAL,
            failure_msg="some failure message",
            relation=lambda x, y: x == y,
        )
    }
    kwargs = {arg1: 1, arg2: 1}
    result, hasError = verify_arg_relations(relations, **kwargs)
    assert len(result) == 0
    assert not hasError


def test_verify_arg_relations_optional_fail():
    arg1 = "arg1"
    arg2 = "arg2"
    relation = ArgRelation(
        level=ParamLevel.OPTIONAL,
        failure_msg="some failure message",
        relation=lambda x, y: x == y,
    )
    relations = {(arg1, arg2): relation}
    kwargs = {arg1: 1, arg2: 2}
    result, hasError = verify_arg_relations(relations, **kwargs)
    assert len(result) == 1
    assert isinstance(result[0], Error)
    assert hasError
    assert result[0].message == relation.failure_msg


@patch("src.common.services.verification.args_missing_msg", wraps=miss_arg)
def test_verify_arg_relations_optional_missing(mock_args_missing_msg):
    arg1 = "arg1"
    arg2 = "arg2"
    relation = ArgRelation(
        level=ParamLevel.OPTIONAL,
        failure_msg="some failure message",
        relation=lambda x, y: x == y,
    )
    relations = {(arg1, arg2): relation}
    kwargs = {arg1: 1}
    result, hasError = verify_arg_relations(relations, **kwargs)
    assert len(result) == 0
    assert not hasError
    assert mock_args_missing_msg.call_count == 0
