import pytest

from pproc.config import types

BASE_OUTPUT = {"target": "fdb", "step": ["0-6", "6-12"]}


@pytest.mark.parametrize(
    "inputs, expected",
    [
        [
            [
                {"stream": "oper", "type": "fc"},
                {"stream": "enfo", "type": "pf", "number": [1, 2, 3]},
            ],
            [
                {**BASE_OUTPUT, "stream": "oper", "type": "gbf"},
                {**BASE_OUTPUT, "stream": "enfo", "type": "gbf", "number": [1, 2, 3]},
                {**BASE_OUTPUT, "stream": "oper", "type": "gwt"},
                {**BASE_OUTPUT, "stream": "enfo", "type": "gwt", "number": [1, 2, 3]},
            ],
        ],
        [
            [
                {"stream": "eefo", "type": "cf"},
                {"stream": "eefo", "type": "pf", "number": [1, 2, 3]},
            ],
            [
                {
                    **BASE_OUTPUT,
                    "stream": "eefo",
                    "type": "gbf",
                    "number": [0, 1, 2, 3],
                },
                {
                    **BASE_OUTPUT,
                    "stream": "eefo",
                    "type": "gwt",
                    "number": [0, 1, 2, 3],
                },
            ],
        ],
        [
            {"stream": "oper", "type": "fc"},
            [
                {**BASE_OUTPUT, "stream": "oper", "type": "gbf"},
                {**BASE_OUTPUT, "stream": "oper", "type": "gwt"},
            ],
        ],
        [
            {"stream": "oper", "type": "fc", "timespan": "fs"},
            [
                {
                    **BASE_OUTPUT,
                    "stream": "oper",
                    "type": "gbf",
                    "timespan": "fs",
                    "step": [6],
                },
                {
                    **BASE_OUTPUT,
                    "stream": "oper",
                    "type": "gbf",
                    "timespan": "6h",
                    "step": [12],
                },
                {
                    **BASE_OUTPUT,
                    "stream": "oper",
                    "type": "gwt",
                    "timespan": "fs",
                    "step": [6],
                },
                {
                    **BASE_OUTPUT,
                    "stream": "oper",
                    "type": "gwt",
                    "timespan": "6h",
                    "step": [12],
                },
            ],
        ],
    ],
)
def test_ecpoint_outputs(inputs, expected):
    config = types.ECPointConfig(
        bp_location="bp.csv",
        fer_location="fer.csv",
        predictant="tp",
        predictors=[],
        parameters={
            "tp": {
                "inputs": {"fc": {"request": inputs}},
                "dependencies": {},
                "num_inputs": 1,
                "accumulations": {"step": {"coords": [[0, 6], [6, 12]]}},
            }
        },
        inputs={"fc": {"source": {"type": "fdb"}}},
        outputs={
            "default": {"target": {"type": "fdb"}},
            "bias_corrected": {"metadata": {"type": "gbf"}},
            "weather_types": {"metadata": {"type": "gwt"}},
            "members": {"target": {"type": "file", "path": "members.grib"}},
        },
    )
    config.print()
    outputs = list(config.out_mars(targets=["fdb"]))
    assert outputs == expected
