import pytest

from pproc.configs import ranges


@pytest.mark.parametrize(
    "date, steps, num_months",
    [
        ["20241001", list(range(24, 5160 + 1, 24)), 7],
        ["20241001", list(range(12, 5160 + 1, 12)), 7],
        ["20241001", list(range(0, 5160 + 1, 6)), 7],
        ["20241002", list(range(6, 5160 + 1, 6)), 6],
        ["20241002", list(range(6, 800, 6)), 0],
    ],
)
def test_monthly(date, steps, num_months):
    step_ranges = ranges.monthly(date, steps)
    assert len(step_ranges) == num_months
    if num_months > 0:
        assert step_ranges[0][0] != 0


@pytest.mark.parametrize(
    "steps, num_weeks, num_steps",
    [
        [list(range(6, 360 + 1, 6)), 8, 29],
        [list(range(0, 1104 + 1, 12)), 40, 15],
        [list(range(360, 5, -6)), 8, 29],
        [["0-168", "24-192"], 2, 1],
    ],
)
def test_weekly(steps, num_weeks, num_steps):
    step_ranges = ranges.weekly(steps)
    print(step_ranges[-1])
    assert len(step_ranges) == num_weeks
    assert all([len(x) == num_steps for x in step_ranges])
