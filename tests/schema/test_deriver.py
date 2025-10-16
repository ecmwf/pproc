import pytest

from pydantic import TypeAdapter

from pproc.schema.deriver import ForecastStepDeriver, ClimStepDeriver, ClimDateDeriver

FcStepDeriver = TypeAdapter(ForecastStepDeriver)


@pytest.mark.parametrize(
    "config, output_request, fc_steps, expected",
    [
        pytest.param({"type_": "default"}, {"step": 6}, [0, 6, 12], [6], id="inst"),
        pytest.param(
            {"type_": "default"}, {"step": "0-6"}, [0, 3, 6, 12], [3, 6], id="range"
        ),
        pytest.param(
            {"type_": "default", "include_start": True},
            {"step": "0-6"},
            [0, 3, 6, 12],
            [0, 3, 6],
            id="range-with-start",
        ),
        pytest.param(
            {"type_": "default", "include_start": True, "allow_missing_zero": True},
            {"step": "0-6"},
            [0, 3, 6, 12],
            [0, 3, 6],
            id="range-allow-missing",
        ),
        pytest.param(
            {"type_": "default", "include_start": True, "allow_missing_zero": True},
            {"step": "0-6"},
            [3, 6, 12],
            [3, 6],
            id="range-missing-zero",
        ),
        pytest.param(
            {"type_": "default", "by": 6},
            {"step": "0-6"},
            [0, 3, 6, 12],
            [6],
            id="range-with-by",
        ),
        pytest.param(
            {"type_": "deaccumulate"},
            {"step": "0-6"},
            [0, 6, 9, 12],
            [0, 6],
            id="deacc-range",
        ),
        pytest.param(
            {"type_": "deaccumulate", "allow_missing_zero": True},
            {"step": "0-6"},
            [6, 9, 12],
            [6],
            id="deacc-missing-zero",
        ),
        pytest.param(
            {"type_": "deaccumulate"},
            {"step": 6},
            [0, 6, 9, 12],
            [0, 6],
            id="deacc-inst",
        ),
        pytest.param(
            {"type_": "deaccumulate", "allow_missing_zero": True},
            {"step": 6},
            [6, 9, 12],
            [6],
            id="deacc-inst-missing-zero",
        ),
        pytest.param(
            {"type_": "deaccumulate", "by": 12},
            {"step": 12},
            [0, 6, 9, 12],
            [0, 12],
            id="deacc-inst-by",
        ),
        pytest.param(
            {"type_": "deaccumulate", "by": 24, "allow_missing_zero": True},
            {"step": "0-12"},
            [12, 18, 24],
            [24],
            id="deacc-range-by",
        ),
        pytest.param(
            {"type_": "precomputed"},
            {"step": "0-12"},
            [6, 9, 12],
            ["0-12"],
            id="precomputed",
        ),
        pytest.param(
            {"type_": "monthly"},
            {"fcmonth": 1, "date": "20250601"},
            list(range(0, 745, 6)),
            list(range(6, 721, 6)),
            id="monthly",
        ),
        pytest.param(
            {"type_": "monthly", "include_start": True},
            {"fcmonth": 1, "date": "20250601"},
            list(range(0, 745, 6)),
            list(range(0, 721, 6)),
            id="monthly-with-start",
        ),
        pytest.param(
            {
                "type_": "monthly",
                "include_start": True,
                "allow_missing_zero": True,
                "by": 24,
            },
            {"fcmonth": 1, "date": "20250601"},
            list(range(6, 745, 6)),
            list(range(24, 721, 24)),
            id="monthly-missing-zero",
        ),
        pytest.param(
            {"type_": "select", "index": 1},
            {"step": "0-24"},
            [0, 12, 24],
            [24],
            id="select",
        ),
        pytest.param(
            {"type_": "select", "index": 0, "include_start": True},
            {"step": "0-24"},
            [0, 12, 24],
            [0],
            id="select-with-start",
        ),
        pytest.param(
            {
                "type_": "select",
                "index": 0,
                "include_start": True,
                "allow_missing_zero": True,
            },
            {"step": "0-24"},
            [12, 24],
            [12],
            id="select-missing-zero",
        ),
        pytest.param(
            {"type_": "static", "values": [0, 12]},
            {"step": "0-24"},
            [0, 12, 24],
            [0, 12],
            id="static",
        ),
    ],
)
def test_forecast_step_deriver(config, output_request, fc_steps, expected):
    deriver = FcStepDeriver.validate_python(config)
    assert expected == deriver.derive(output_request, fc_steps)


@pytest.mark.parametrize(
    "step_type, times, step, clim_steps, expected",
    [
        [
            "instantaneous",
            ["0000", "0600"],
            [12, 24, 360],
            list(range(0, 361, 12)),
            [12, 24, 360],
        ],
        [
            "instantaneous",
            ["1200", "1800"],
            [12, 24, 360],
            list(range(0, 361, 12)),
            [24, 36, 348],
        ],
        [
            "range",
            ["0000", "0600"],
            list(range(0, 25, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "0-24",
        ],
        [
            "range",
            ["0000", "0600"],
            list(range(0, 241, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "0-240",
        ],
        [
            "range",
            ["0000", "0600"],
            list(range(12, 25, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "0-24",
        ],
        [
            "range",
            ["0000", "0600"],
            [24],
            ["0-24", "12-36", "24-28", "0-240"],
            "0-24",
        ],
        [
            "range",
            ["1200", "1800"],
            list(range(0, 25, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "12-36",
        ],
        [
            "range",
            ["1200", "1800"],
            list(range(0, 241, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "0-240",
        ],
        [
            "range",
            ["1200", "1800"],
            list(range(12, 25, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "12-36",
        ],
        [
            "range",
            ["1200", "1800"],
            [240],
            ["0-24", "12-36", "24-28", "0-240"],
            "0-240",
        ],
        ["range", ["0000"], ["0-168"], ["0-168", "24-192", "48-216"], "0-168"],
        ["range", ["0000"], ["24-192"], ["0-168", "24-192", "48-216"], "24-192"],
    ],
)
def test_clim_step_deriver(step_type, times, step, clim_steps, expected):
    deriver = ClimStepDeriver(type=step_type)
    for time in times:
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
