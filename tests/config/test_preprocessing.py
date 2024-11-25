from contextlib import nullcontext
from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
from pydantic import ValidationError
import pytest

from pproc.config.preprocessing import (
    Combination,
    MaskExpression,
    Masking,
    Preprocessing,
    PreprocessingConfig,
    Scaling,
)


_MASKEXPR_LEN_ERR = r"Mask expression should be a \[lhs, cmp, rhs\] list"


@pytest.mark.parametrize(
    "inp, err",
    [
        pytest.param([], _MASKEXPR_LEN_ERR, id="empty"),
        pytest.param([1.0], _MASKEXPR_LEN_ERR, id="tooshort"),
        pytest.param([2, "<", {}, "<", 3], _MASKEXPR_LEN_ERR, id="toolong"),
        pytest.param(
            [1.0, 2.0, 3.0],
            "Input should be '<', '>', '>=', '<=', '==' or '!='",
            id="nocmp",
        ),
        pytest.param(
            ["foo", "!=", 1], "Input should be a valid number", id="invalid-val"
        ),
        pytest.param([{}, ">=", 0], None, id="left-field"),
        pytest.param([{"param": 1}, ">", {"param": 2}], None, id="both-fields"),
        pytest.param([12.3, "<=", {"levelist": 2, "param": 4}], None, id="right-field"),
    ],
)
def test_maskexpr_fromlist(inp: list, err: Optional[str]):
    ctx = nullcontext() if err is None else pytest.raises(ValidationError, match=err)
    with ctx:
        expr = MaskExpression.model_validate(inp)
    if err is None:
        assert expr.lhs == inp[0]
        assert expr.cmp == inp[1]
        assert expr.rhs == inp[2]


@pytest.mark.parametrize(
    "inp, exp",
    [
        pytest.param([{"param": 3}, ">", 1.0], None, id="notfound-left"),
        pytest.param([2.3, ">", {"param": 1, "step": 12}], None, id="notfound-right"),
        pytest.param([{"param": 1}, ">", 1.0], [0, 1, 1], id="left-field->"),
        pytest.param(
            [{"param": 1}, "<=", {"param": 2}], [0, 1, 1], id="both-fields-<="
        ),
        pytest.param([2, "<", {"param": 1}], [0, 0, 1], id="right-field-<"),
        pytest.param([{"param": 2}, ">=", 3.0], [0, 1, 1], id="left-field->="),
        pytest.param([{"levelist": 5}, "==", 0.0], [1, 0, 0], id="left-field-=="),
        pytest.param([2, "!=", {"param": 1}], [1, 0, 1], id="right-field-!="),
    ],
)
def test_maskexpr_apply(inp: list, exp: Optional[list]):
    expr = MaskExpression.model_validate(inp)
    ctx = (
        pytest.raises(KeyError, match="No data matching")
        if exp is None
        else nullcontext()
    )
    with ctx:
        result = expr.apply(
            [{"param": 1}, {"param": 2, "levelist": 5}],
            [np.array([1.0, 2.0, 3.0]), np.array([0.0, 3.0, 3.0])],
        )
    if exp is not None:
        expected = np.array(exp)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "value, inp", [(1.0, [1, 2, 3]), (0.0, [2, 5, 3, 4]), (5.0, [1, 2])]
)
def test_scaling_apply(value: float, inp: list):
    scaling = Scaling(value=value)
    metadata = [{"param": 1}, {"param": 2, "levelist": 4}]
    inp_arr = np.array(inp)
    data = [inp_arr + i for i in range(len(metadata))]
    new_metadata, new_data = scaling.apply(metadata, data)
    assert new_metadata == metadata
    np.testing.assert_array_equal(
        new_data, [value * (inp_arr + i) for i in range(len(metadata))]
    )


@pytest.mark.parametrize(
    "op, arrs, exp",
    [
        pytest.param(
            "direction",
            [[1, 2]],
            "'direction' requires exactly 2 input fields",
            id="dir-tooshort",
        ),
        pytest.param(
            "direction",
            [[1, 2], [3, 4], [5, 6]],
            "'direction' requires exactly 2 input fields",
            id="dir-toolong",
        ),
        pytest.param("direction", [[0, 1], [1, 0]], [180, 270], id="dir"),
        pytest.param("norm", [[1, 2, 4]], [1, 2, 4], id="norm-1"),
        pytest.param("norm", [[0, 1, 3], [1, 0, 4]], [1, 1, 5], id="norm-2"),
        pytest.param(
            "norm", [[1, 1, 3], [1, 0, 0], [1, 0, 4], [1, 0, 0]], [2, 1, 5], id="norm-4"
        ),
        pytest.param("sum", [[1, 3]], [1, 3], id="sum-1"),
        pytest.param("sum", [[1, 2, 6], [3, 8, 3]], [4, 10, 9], id="sum-2"),
        pytest.param("sum", [[0, -1, 1], [1, 0, -1], [2, 2, 0]], [3, 1, 0], id="sum-3"),
    ],
)
def test_combination_apply(
    op: Literal["direction", "norm", "sum"], arrs: List[list], exp: Union[list, str]
):
    combination = Combination(operation=op)
    metadata = [{"param": i} for i in range(len(arrs))]
    data = [np.array(arr) for arr in arrs]
    ctx = (
        pytest.raises(AssertionError, match=exp)
        if isinstance(exp, str)
        else nullcontext()
    )
    with ctx:
        new_metadata, new_data = combination.apply(metadata, data)
    if not isinstance(exp, str):
        assert new_metadata == [metadata[0]]
        np.testing.assert_array_equal(new_data, [exp])


@pytest.mark.parametrize(
    "mask, select, replacement, exp",
    [
        pytest.param(
            [{"param": 5}, "<", 1.0], {"param": 1}, 0.0, None, id="lhs-missing"
        ),
        pytest.param(
            [{"param": 1}, ">", 2.0], {"param": 6}, 0.0, None, id="sel-missing"
        ),
        pytest.param(
            [{"param": 2}, "==", {"param": 4}],
            {"param": 1},
            0.0,
            None,
            id="rhs-missing",
        ),
        pytest.param(
            [{"param": 1}, ">=", 2.0],
            {"param": 1},
            -1.0,
            (0, [1, -1, -1]),
            id="single-param",
        ),
        pytest.param(
            [{"param": 2}, "<", 3.0], {"param": 1}, 4.0, (0, [4, 4, 3]), id="two-params"
        ),
        pytest.param(
            [{"param": 1}, ">", {"param": 3}],
            {"levelist": 4},
            0.0,
            (1, [0, 2, 0]),
            id="three-params",
        ),
    ],
)
def test_masking_apply(
    mask: list, select: dict, replacement: float, exp: Optional[Tuple[int, list]]
):
    masking = Masking(
        mask=MaskExpression.model_validate(mask), select=select, replacement=replacement
    )
    metadata = [{"param": 1}, {"param": 2, "levelist": 4}, {"param": 3}]
    data = [np.array([1, 2, 3]), np.array([0, 2, 5]), np.array([1, 3, 1])]
    ctx = (
        pytest.raises(KeyError, match="No data matching")
        if exp is None
        else nullcontext()
    )
    with ctx:
        new_metadata, new_data = masking.apply(metadata, data)
    if exp is not None:
        assert new_metadata == [metadata[exp[0]]]
        np.testing.assert_array_equal(new_data, [exp[1]])


@pytest.mark.parametrize(
    "proc, exp",
    [
        pytest.param(
            Scaling(value=2.0, output={"type": "fcmean"}),
            [
                {"param": 1, "type": "fcmean"},
                {"param": 2, "levelist": 4, "type": "fcmean"},
                {"param": 3, "type": "fcmean"},
            ],
            id="scaling",
        ),
        pytest.param(
            Combination(operation="sum", output={"param": 4}),
            [{"param": 4}],
            id="combination",
        ),
        pytest.param(
            Masking(
                mask=MaskExpression.model_validate([{"param": 2}, ">", 0.0]),
                select={"param": 3},
                output={"param": 31},
            ),
            [{"param": 31, "type": "em"}],
            id="masking",
        ),
    ],
)
def test_preprocessing_output(proc: Preprocessing, exp: List[dict]):
    metadata = [{"param": 1}, {"param": 2, "levelist": 4}, {"param": 3, "type": "em"}]
    data = [np.array([1, 2, 3]), np.array([0, 2, 5]), np.array([1, 3, 1])]
    new_metadata, _ = proc.apply(metadata, data)
    assert new_metadata == exp


@pytest.mark.parametrize(
    "inp, exp",
    [
        pytest.param([], [], id="empty"),
        pytest.param([{"operation": "scale", "value": 10}], [Scaling], id="scale"),
        pytest.param(
            [
                {"operation": "sum"},
                {"operation": "mask", "mask": [{}, ">=", 5], "select": {}},
            ],
            [Combination, Masking],
            id="comb-mask",
        ),
        pytest.param(
            [
                {"operation": "mask", "mask": [3, "!=", {}], "select": {}},
                {"operation": "scale", "value": 5},
                {"operation": "mask", "mask": [{}, ">", 10], "select": {}},
            ],
            [Masking, Scaling, Masking],
            id="mask-scale-mask",
        ),
    ],
)
def test_preprocessing_config_fromlist(inp: List[dict], exp: List[Type[Preprocessing]]):
    preproc = PreprocessingConfig.model_validate(inp)
    assert len(preproc.actions) == len(exp)
    for action, exp_type in zip(preproc.actions, exp):
        assert type(action) is exp_type


@pytest.mark.parametrize(
    "inp, num, exp",
    [
        pytest.param([], 1, [1, 4, 9, 16], id="empty"),
        pytest.param([{"operation": "sum"}], 3, [6, 13, 23, 35], id="sum"),
        pytest.param(
            [
                {
                    "operation": "mask",
                    "select": {"param": 1},
                    "mask": [{"param": 2}, ">=", 5],
                    "replacement": -1,
                },
                {"operation": "scale", "value": 2},
            ],
            2,
            [2, 8, -2, -2],
            id="mask-scale",
        ),
    ],
)
def test_preprocessing_config_apply(inp: List[dict], num: int, exp: list):
    preproc = PreprocessingConfig.model_validate(inp)
    metadata = [{"param": 1}, {"param": 2}, {"param": 3}]
    data = [np.array([1, 4, 9, 16]), np.array([2, 3, 5, 7]), np.array([3, 6, 9, 12])]
    _, new_data = preproc.apply(metadata[:num], data[:num])
    np.testing.assert_equal(new_data, [exp])
