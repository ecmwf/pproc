import pytest

from pproc.configs import ranges


@pytest.mark.parametrize(
    "date, steps, num_months, remaining",
    [
        ["20241001", list(range(24, 5160 + 1, 24)), 7, 3],
        ["20241001", list(range(12, 5160 + 1, 12)), 7, 6],
        ["20241001", list(range(6, 5160 + 1, 6)), 7, 12],
        ["20241001", list(range(0, 5160 + 1, 6)), 7, 12],
        ["20241002", list(range(6, 5160 + 1, 6)), 6, 16],
        ["20241002", list(range(6, 800, 6)), 0, 0],
    ],
)
def test_monthly(date, steps, num_months, remaining):
    step_ranges = ranges.monthly(date, steps)
    assert len(step_ranges) == num_months
    if num_months > 0:
        assert step_ranges[0]["from"] != 0
        assert len(steps[steps.index(step_ranges[-1]["to"]) + 1 :]) == remaining


@pytest.mark.parametrize(
    "steps, num_weeks, num_steps",
    [
        [list(range(6, 360 + 1, 6)), 8, 29],
        [list(range(0, 360 + 1, 6)), 9, 29],
        [list(range(0, 1104 + 1, 12)), 40, 15],
        [list(range(360, 5, -6)), 8, 29],
    ],
)
def test_weekly(steps, num_weeks, num_steps):
    step_ranges = ranges.weekly("20240909", steps)
    assert len(step_ranges) == num_weeks
    for x in step_ranges:
        if isinstance(x, dict):
            assert len(range(x["from"], x["to"] + 1, x["by"])) == num_steps
        else:
            assert len(x) == num_steps


@pytest.mark.parametrize(
    "accum, req, expected",
    [
        [
            {"step": {"range_type": "monthly"}},
            {"date": 20241001, "step": list(range(0, 5161, 6))},
            7,
        ],
        [
            {"step": {"range_type": "weekly"}},
            {"date": 20241001, "step": list(range(0, 361, 6))},
            9,
        ],
        [{"step": {}}, {"step": ["0-168", "24-196"]}, 2],
        [{"hdate": {}}, {"hdate": ["20020101", "20030101"]}, 2],
    ],
)
def test_populate_accums(accum, req, expected):
    ranges.populate_accums(accum, req)
    for accum_config in accum.values():
        assert len(accum_config["coords"]) == expected
