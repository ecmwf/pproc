import numpy as np

from pproc.common import (
    DiffWindow,
    SimpleOpWindow,
    WeightedSumWindow,
    DiffDailyRateWindow,
    Window,
)


def test_instantaneous_window():
    window = Window({"range": [1, 1]}, True)
    step_values = np.array([[1, 1, 1], [2, 2, 2]])
    window.add_step_values(0, step_values)
    assert len(window.step_values) == 0
    window.add_step_values(1, step_values)
    assert np.all(window.step_values == step_values)


def test_period_min():
    window = SimpleOpWindow({"range": [0, 2]}, "min", False)
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    window.add_step_values(0, step_values)
    window.add_step_values(1, step_values)
    window.add_step_values(2, [[2, 4, 6], [1, 2, 3]])
    assert np.all(window.step_values == [[1, 2, 3], [1, 2, 3]])


def test_period_max():
    window = SimpleOpWindow({"range": [0, 2]}, "max", False)
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    window.add_step_values(1, step_values)
    window.add_step_values(2, [[2, 4, 6], [1, 2, 3]])
    assert np.all(window.step_values == [[2, 4, 6], [2, 4, 6]])


def test_period_sum():
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


def test_period_diff():
    window = DiffWindow({"range": [0, 2]})
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    window.add_step_values(0, step_values)
    window.add_step_values(1, step_values)
    window.add_step_values(2, step_values * 2)
    assert np.all(window.step_values == (step_values))


def test_period_weighted_sum():
    window = WeightedSumWindow({"range": [0, 2]})
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    window.add_step_values(1, step_values)
    window.add_step_values(2, step_values * 2)
    assert np.all(window.step_values == (step_values + step_values * 2) / 2)


def test_period_diff_daily_rate():
    window = DiffDailyRateWindow({"range": [0, 240]})
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    window.add_step_values(0, step_values)
    window.add_step_values(120, step_values)
    window.add_step_values(240, step_values * 2)
    assert np.all(window.step_values == (step_values / 10))
