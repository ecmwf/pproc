import pytest

from pproc.configs.ranges import monthly

@pytest.mark.parametrize(
        "date, steps, num_months", [
            ["20241001", list(range(24, 5160 + 1, 24)), 7],
            ["20241001", list(range(12, 5160 + 1, 12)), 7],
            ["20241001", list(range(0, 5160 + 1, 6)), 7],
            ["20241002", list(range(6, 5160 + 1, 6)), 6],
            ["20241002", list(range(6, 800, 6)), 0],
        ]
)
def test_monthly(date, steps, num_months):
    ranges = monthly(date, steps)
    assert len(ranges) == num_months
    if num_months > 0:
        assert ranges[0][0] != 0