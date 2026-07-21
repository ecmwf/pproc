import pytest
from pydantic import TypeAdapter

from ppcore.schema.forecast import Dataset, Climatology
from ppcore.schema.deriver import ClimDateDeriver, ClimatologyDeriver, ForecastDeriver

FcDeriver = TypeAdapter(ForecastDeriver)
ClimDeriver = TypeAdapter(ClimatologyDeriver)


@pytest.mark.parametrize(
    "config, output_request, fc_steps, expected",
    [
        pytest.param({"name": "step_default"}, {"step": 6}, [0, 6, 12], [6], id="inst"),
        pytest.param(
            {"name": "step_default"}, {"step": "0-6"}, [0, 3, 6, 12], [3, 6], id="range"
        ),
        pytest.param(
            {"name": "step_default", "include_start": True},
            {"step": "0-6"},
            [0, 3, 6, 12],
            [0, 3, 6],
            id="range-with-start",
        ),
        pytest.param(
            {"name": "step_default", "include_start": True, "allow_missing_zero": True},
            {"step": "0-6"},
            [0, 3, 6, 12],
            [0, 3, 6],
            id="range-allow-missing",
        ),
        pytest.param(
            {"name": "step_default", "include_start": True, "allow_missing_zero": True},
            {"step": "0-6"},
            [3, 6, 12],
            [3, 6],
            id="range-missing-zero",
        ),
        pytest.param(
            {"name": "step_default", "by": 6},
            {"step": "0-6"},
            [0, 3, 6, 12],
            [6],
            id="range-with-by",
        ),
        pytest.param(
            {"name": "step_deaccumulate"},
            {"step": "0-6"},
            [0, 6, 9, 12],
            [0, 6],
            id="deacc-range",
        ),
        pytest.param(
            {"name": "step_deaccumulate", "allow_missing_zero": True},
            {"step": "0-6"},
            [6, 9, 12],
            [6],
            id="deacc-missing-zero",
        ),
        pytest.param(
            {"name": "step_deaccumulate"},
            {"step": 6},
            [0, 6, 9, 12],
            [0, 6],
            id="deacc-inst",
        ),
        pytest.param(
            {"name": "step_deaccumulate", "allow_missing_zero": True},
            {"step": 6},
            [6, 9, 12],
            [6],
            id="deacc-inst-missing-zero",
        ),
        pytest.param(
            {"name": "step_deaccumulate", "by": 12},
            {"step": 12},
            [0, 6, 9, 12],
            [0, 12],
            id="deacc-inst-by",
        ),
        pytest.param(
            {"name": "step_deaccumulate", "by": 24, "allow_missing_zero": True},
            {"step": "0-12"},
            [12, 18, 24],
            [24],
            id="deacc-range-by",
        ),
        pytest.param(
            {"name": "step_precomputed"},
            {"step": "0-12"},
            [6, 9, 12],
            ["0-12"],
            id="precomputed",
        ),
        pytest.param(
            {"name": "step_monthly"},
            {"fcmonth": 1, "date": "20250601"},
            list(range(0, 745, 6)),
            list(range(6, 721, 6)),
            id="monthly",
        ),
        pytest.param(
            {"name": "step_monthly", "include_start": True},
            {"fcmonth": 1, "date": "20250601"},
            list(range(0, 745, 6)),
            list(range(0, 721, 6)),
            id="monthly-with-start",
        ),
        pytest.param(
            {
                "name": "step_monthly",
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
            {"name": "step_select", "index": 1},
            {"step": "0-24"},
            [0, 12, 24],
            [24],
            id="select",
        ),
        pytest.param(
            {"name": "step_select", "index": 0, "include_start": True},
            {"step": "0-24"},
            [0, 12, 24],
            [0],
            id="select-with-start",
        ),
        pytest.param(
            {
                "name": "step_select",
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
            {"name": "step_static", "values": [0, 12]},
            {"step": "0-24"},
            [0, 12, 24],
            [0, 12],
            id="static",
        ),
    ],
)
def test_forecast_step_deriver(config, output_request, fc_steps, expected):
    deriver = FcDeriver.validate_python(config)
    assert expected == deriver.derive(
        output_request, Dataset(datacubes=[{"step": fc_steps}])
    )


@pytest.mark.parametrize(
    "step_type, times, step, clim_steps, expected",
    [
        [
            "clim_step_inst",
            ["0000"],
            [0, 6, 12, 18, 24, 354, 360],
            list(range(0, 361, 6)),
            [0, 6, 12, 18, 24, 354, 360],
        ],
        [
            "clim_step_inst",
            ["0600"],
            [0, 6, 12, 18, 24, 138],
            list(range(0, 361, 6)),
            [6, 12, 18, 24, 30, 144],
        ],
        [
            "clim_step_inst",
            ["1200"],
            [0, 6, 12, 18, 24, 354, 360],
            list(range(0, 361, 6)),
            [12, 18, 24, 30, 36, 342, 348],
        ],
        [
            "clim_step_inst",
            ["1800"],
            [0, 6, 12, 18, 24, 138],
            list(range(0, 361, 6)),
            [18, 24, 30, 36, 42, 156],
        ],
        [
            "clim_step_range",
            ["0000", "0600"],
            list(range(0, 25, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "0-24",
        ],
        [
            "clim_step_range",
            ["0000", "0600"],
            list(range(0, 241, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "0-240",
        ],
        [
            "clim_step_range",
            ["0000", "0600"],
            list(range(12, 25, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "0-24",
        ],
        [
            "clim_step_range",
            ["0000", "0600"],
            [24],
            ["0-24", "12-36", "24-28", "0-240"],
            "0-24",
        ],
        [
            "clim_step_range",
            ["1200", "1800"],
            list(range(0, 25, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "12-36",
        ],
        [
            "clim_step_range",
            ["1200", "1800"],
            list(range(0, 241, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "0-240",
        ],
        [
            "clim_step_range",
            ["1200", "1800"],
            list(range(12, 25, 12)),
            ["0-24", "12-36", "24-28", "0-240"],
            "12-36",
        ],
        [
            "clim_step_range",
            ["1200", "1800"],
            [240],
            ["0-24", "12-36", "24-28", "0-240"],
            "0-240",
        ],
        [
            "clim_step_range",
            ["0000"],
            ["0-168"],
            ["0-168", "24-192", "48-216"],
            "0-168",
        ],
        [
            "clim_step_range",
            ["0000"],
            ["24-192"],
            ["0-168", "24-192", "48-216"],
            "24-192",
        ],
    ],
)
def test_clim_step_deriver(step_type, times, step, clim_steps, expected):
    deriver = ClimDeriver.validate_python({"name": step_type})  # type: ignore
    for time in times:
        assert (
            deriver.derive(
                {"time": time, "step": step},
                Climatology(datacubes=[{"step": clim_steps}], scheme="ecmwf-4days"),
            )
            == expected
        )


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
    assert (
        deriver.derive({"date": date}, Climatology(datacubes=[], scheme="ecmwf-2days"))
        == expected
    )
