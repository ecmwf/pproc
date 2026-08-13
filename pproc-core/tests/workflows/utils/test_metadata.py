import pytest

from earthkit.workflows.plugins.pproc.utils.metadata import fill_template_value


@pytest.mark.parametrize(
    "value, template_map, expected",
    [
        ("{start_coord}", {"start_coord": "20240507"}, "20240507"),
        ("{coords_span}:int", {"coords_span": 24}, 24),
        ("{start_coord[0:6]}", {"start_coord": "20240507"}, "202405"),
        ("{int(start_coord[0:6])}", {"start_coord": "20240507"}, 202405),
        ("{int(start_coord[0:4]) - 20}", {"start_coord": "20240507"}, 2004),
        (
            "clim_{start_coord[0:4]}",
            {"start_coord": "20240507"},
            "clim_{start_coord[0:4]}",
        ),
        (
            "{start_coord[0:4]}",
            {"end_coord": "20240507"},
            "{start_coord[0:4]}",
        ),
    ],
    ids=[
        "no-type",
        "typed-int",
        "slice-str",
        "slice-int",
        "slice-expression",
        "noop",
        "missing-key",
    ],
)
def test_fill_template_value(value, template_map, expected):
    assert fill_template_value(value, template_map) == expected
