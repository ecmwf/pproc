import numpy as np
import pytest

from pproc.common.window import create_window, parse_window_config


@pytest.mark.parametrize(
    "steps, include_init, exp",
    [
        pytest.param([0, 0], True, [0], id="0-0"),
        pytest.param([3, 10], True, list(range(3, 11)), id="3-10"),
        pytest.param([2, 6], False, list(range(3, 7)), id="2-6-noinit"),
        pytest.param([3, 12, 3], True, [3, 6, 9, 12], id="3-12-by3"),
        pytest.param([1, 1, 6], True, [1], id="1-1-by6"),
        pytest.param([0, 1, 4], True, [0], id="0-1-by4"),
        pytest.param([0, 24, 8], False, [8, 16, 24], id="0-24-by8-noinit"),
    ],
)
def test_window_steps(steps, include_init, exp):
    window = parse_window_config({"range": steps}, include_init)
    assert window.steps == exp


def test_instantaneous_window():
    accum = create_window({"range": [1, 1]}, "none", True)
    step_values = np.array([[1, 1, 1], [2, 2, 2]])
    accum.feed(0, step_values)
    assert accum.get_values() is None
    accum.feed(1, step_values)
    values = accum.get_values()
    assert values is not None
    np.testing.assert_equal(values, step_values)


@pytest.mark.parametrize(
    "window_operation, values",
    [
        ["minimum", [[1, 2, 3], [1, 2, 3]]],
        ["maximum", [[2, 4, 6], [2, 4, 6]]],
        ["add", [[3, 6, 9], [3, 6, 9]]],
    ],
)
def test_simple_op(window_operation, values):
    accum = create_window({"range": [0, 2]}, window_operation, False)
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    accum.feed(0, step_values)
    accum.feed(1, step_values)
    accum.feed(2, [[2, 4, 6], [1, 2, 3]])
    acc_values = accum.get_values()
    assert acc_values is not None
    np.testing.assert_equal(acc_values, values)


def test_multi_windows():
    accum = create_window({"range": [0, 2]}, "add", False)
    accum2 = create_window({"range": [0, 2]}, "add", False)
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    accum.feed(1, step_values)
    accum2.feed(1, step_values)
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    accum.feed(2, step_values * 2)
    values = accum.get_values()
    assert values is not None
    np.testing.assert_equal(values, np.array([[3, 6, 9], [6, 12, 18]]))
    accum2.feed(2, step_values)
    values2 = accum2.get_values()
    assert values2 is not None
    np.testing.assert_equal(values2, np.array([[2, 4, 6], [4, 8, 12]]))


@pytest.mark.parametrize(
    "operation, include_init, end_step, step_increment, values",
    [
        pytest.param("diff", True, 2, 1, [[1, 2, 3], [2, 4, 6]], id="diff"),
        pytest.param(
            "weightedsum", False, 2, 1, [[1.5, 3, 4.5], [3, 6, 9]], id="weightedsum"
        ),
        pytest.param(
            "diffdailyrate",
            True,
            240,
            120,
            np.divide([[1, 2, 3], [2, 4, 6]], 10),
            id="diffdailyrate",
        ),
        pytest.param("mean", False, 6, 3, [[1.5, 3, 4.5], [3, 6, 9]], id="mean"),
    ],
)
def test_windows(operation, include_init, end_step, step_increment, values):
    accum = create_window({"range": [0, end_step]}, operation, include_init)
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    accum.feed(0, step_values)
    accum.feed(step_increment, step_values)
    accum.feed(2 * step_increment, step_values * 2)
    acc_values = accum.get_values()
    assert acc_values is not None
    np.testing.assert_almost_equal(acc_values, values)


@pytest.mark.parametrize(
    "start_end, operation, extra_keys, grib_key_values",
    [
        pytest.param(
            [1, 1], "none", None, {"step": "1", "timeRangeIndicator": 0}, id="inst"
        ),
        pytest.param(
            [0, 0], "none", None, {"step": "0", "timeRangeIndicator": 1}, id="inst-0"
        ),
        pytest.param(
            [260, 260],
            "none",
            None,
            {"step": "260", "timeRangeIndicator": 10},
            id="inst-260",
        ),
        pytest.param(
            [1, 2], "maximum", None, {"stepRange": "1-2", "stepType": "max"}, id="range"
        ),
        pytest.param(
            [320, 360],
            "maximum",
            None,
            {"stepRange": "320-360", "stepType": "max", "unitOfTimeRange": 11},
            id="range-360",
        ),
        pytest.param(
            [1, 2],
            "mean",
            {"stepType": "avg", "numberIncludedInAverage": 2},
            {"numberIncludedInAverage": 2, "stepRange": "1-2", "stepType": "avg"},
            id="extra",
        ),
    ],
)
def test_grib_header(start_end, operation, extra_keys, grib_key_values):
    accum = create_window({"range": start_end}, operation, True, extra_keys)
    header = accum.grib_keys()
    assert header == grib_key_values
