import numpy as np
import pytest

from pproc.common import (
    DiffWindow,
    SimpleOpWindow,
    WeightedSumWindow,
    DiffDailyRateWindow,
    Window,
    MeanWindow,
)


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
    window = Window({"range": steps}, include_init)
    assert window.steps == exp


def test_instantaneous_window():
    window = Window({"range": [1, 1]}, True)
    step_values = np.array([[1, 1, 1], [2, 2, 2]])
    window.add_step_values(0, step_values)
    assert len(window.step_values) == 0
    window.add_step_values(1, step_values)
    assert np.all(window.step_values == step_values)


@pytest.mark.parametrize(
    "window_operation, values",
    [
        ["min", [[1, 2, 3], [1, 2, 3]]],
        ["max", [[2, 4, 6], [2, 4, 6]]],
        ["sum", [[3, 6, 9], [3, 6, 9]]],
        ["concatenate", [[1, 2, 3], [2, 4, 6], [2, 4, 6], [1, 2, 3]]],
    ],
)
def test_simple_op(window_operation, values):
    window = SimpleOpWindow({"range": [0, 2]}, window_operation, False)
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    window.add_step_values(0, step_values)
    window.add_step_values(1, step_values)
    window.add_step_values(2, [[2, 4, 6], [1, 2, 3]])
    assert np.all(window.step_values == values)


def test_multi_windows():
    window = SimpleOpWindow({"range": [0, 2]}, "sum", False)
    window2 = SimpleOpWindow({"range": [0, 2]}, "sum", False)
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    window.add_step_values(1, step_values)
    window2.add_step_values(1, step_values)
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    window.add_step_values(2, step_values * 2)
    assert np.all(window.step_values == np.array([[3, 6, 9], [6, 12, 18]]))
    window2.add_step_values(2, step_values)
    assert np.all(window2.step_values == np.array([[2, 4, 6], [4, 8, 12]]))


@pytest.mark.parametrize(
    "window_class, end_step, step_increment, values",
    [
        [DiffWindow, 2, 1, [[1, 2, 3], [2, 4, 6]]],
        [WeightedSumWindow, 2, 1, [[1.5, 3, 4.5], [3, 6, 9]]],
        [DiffDailyRateWindow, 240, 120, np.divide([[1, 2, 3], [2, 4, 6]], 10)],
        [MeanWindow, 6, 3, [[1.5, 3, 4.5], [3, 6, 9]]],
    ],
)
def test_window_classes(window_class, end_step, step_increment, values):
    window = window_class({"range": [0, end_step]})
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    window.add_step_values(0, step_values)
    window.add_step_values(step_increment, step_values)
    window.add_step_values(2 * step_increment, step_values * 2)
    assert np.all(window.step_values == values)


@pytest.mark.parametrize(
    "start_end, grib_key_values",
    [
        [[1, 1], {"step": "1"}],
        [[1, 2], {"stepRange": "1-2", "stepType": "max"}],
        [
            [320, 360],
            {"stepRange": "320-360", "stepType": "max", "unitOfTimeRange": 11},
        ],
    ],
)
def test_grib_header(start_end, grib_key_values):
    window = Window({"range": start_end}, True)
    header = window.grib_header()
    assert header.keys() == grib_key_values.keys()
    for key, value in grib_key_values.items():
        assert header[key] == value
