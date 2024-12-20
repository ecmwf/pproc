import pytest 

from pproc.common.stepseq import stepseq_monthly

@pytest.mark.parametrize(
    "date, start, end, interval, num_months, remaining",
    [
        ["20241001", 12, 5160, 12, 7, 6],
        ["20241001", 6, 5160, 6, 7, 12],
        ["20241001", 0, 5160, 6, 7, 12],
        ["20241002", 6, 5160, 6, 6, 16],
        ["20241002", 6, 800, 6, 0, 0],
    ],
)
def test_monthly(date, start, end, interval, num_months, remaining):
    steps = list(range(start, end + 1, interval))
    step_ranges = [x for x in stepseq_monthly(date, start, end, interval)]
    assert len(step_ranges) == num_months
    if num_months > 0:
        assert len(steps[steps.index(step_ranges[-1][-1]) + 1 :]) == remaining