import pytest

from pproc.schema.deriver import ClimDateDeriver


@pytest.mark.parametrize(
    "config, date, expected",
    [
        [{"option": "next", "sequence": "daily"}, "20250930", "20251001"],
        [{"option": "previous", "strict": False}, "20250930", "20250929"],
        [{"option": "previous"}, "20250929", "20250927"],
        [{"option": "nearest"}, "20250930", "20250929"],
        [
            {"option": "bracket", "num": 1, "strict": False},
            "20250930",
            ["20250929", "20251001"],
        ],
    ],
)
def test_clim_date_deriver(config, date, expected):
    deriver = ClimDateDeriver(**config)
    assert deriver.derive({"date": date}, "ecmwf-2days") == expected
