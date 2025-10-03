import pytest

from pproc.schema.deriver import ClimStepDeriver, ClimDateDeriver


@pytest.mark.parametrize(
    "step_type, time, step, clim_steps, expected",
    [
        [
            "instantaneous",
            "0000",
            [12, 24, 360],
            list(range(0, 361, 12)),
            [12, 24, 360],
        ],
        [
            "instantaneous",
            "0600",
            [12, 24, 360],
            list(range(0, 361, 12)),
            [12, 24, 360],
        ],
        [
            "instantaneous",
            "1200",
            [12, 24, 360],
            list(range(0, 361, 12)),
            [24, 36, 348],
        ],
        [
            "instantaneous",
            "1800",
            [12, 24, 360],
            list(range(0, 361, 12)),
            [24, 36, 348],
        ],
        [
            "range",
            "0000",
            list(range(0, 25, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "0-24",
        ],
        [
            "range",
            "0000",
            list(range(0, 241, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "0-240",
        ],
        [
            "range",
            "0600",
            list(range(0, 25, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "0-24",
        ],
        [
            "range",
            "0600",
            list(range(0, 241, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "0-240",
        ],
        [
            "range",
            "1200",
            list(range(0, 25, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "12-36",
        ],
        [
            "range",
            "1200",
            list(range(0, 241, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "0-240",
        ],
        [
            "range",
            "1800",
            list(range(0, 25, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "12-36",
        ],
        [
            "range",
            "1800",
            list(range(0, 241, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "0-240",
        ],
        ["range", "0000", ["0-168"], ["0-168", "24-192", "48-216"], "0-168"],
        ["range", "0000", ["24-192"], ["0-168", "24-192", "48-216"], "24-192"],
    ],
)
def test_clim_step_deriver(step_type, time, step, clim_steps, expected):
    deriver = ClimStepDeriver(type=step_type)
    assert deriver.derive({"time": time, "step": step}, clim_steps) == expected


@pytest.mark.parametrize(
    "config, date, expected",
    [
        [{"option": "next", "sequence": {"type": "daily"}}, "20250930", "20251001"],
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
