import pytest

from pproc.config import types

BASE_OUTPUT = {
    "method": "ecPoint",
    "target": "fdb",
}


@pytest.mark.parametrize(
    "inputs, perc_metadata, expected",
    [
        [
            [
                {"stream": "oper", "type": "fc"},
                {"stream": "enfo", "type": "pf", "number": [1, 2, 3]},
            ],
            {"stream": "enfo"},
            [
                {**BASE_OUTPUT, "stream": "oper", "type": "gbf"},
                {**BASE_OUTPUT, "stream": "enfo", "type": "gbf", "number": [1, 2, 3]},
                {**BASE_OUTPUT, "stream": "oper", "type": "gwt"},
                {**BASE_OUTPUT, "stream": "enfo", "type": "gwt", "number": [1, 2, 3]},
                {
                    **BASE_OUTPUT,
                    "stream": "enfo",
                    "type": "pfc",
                    "quantile": [f"{x}:{100}" for x in range(1, 101)],
                },
            ],
        ],
        [
            [
                {"stream": "eefo", "type": "cf"},
                {"stream": "eefo", "type": "pf", "number": [1, 2, 3]},
            ],
            {},
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
                {
                    **BASE_OUTPUT,
                    "stream": "eefo",
                    "type": "pfc",
                    "quantile": [f"{x}:{100}" for x in range(1, 101)],
                },
            ],
        ],
        [
            {"stream": "oper", "type": "fc"},
            {},
            [
                {**BASE_OUTPUT, "stream": "oper", "type": "gbf"},
                {**BASE_OUTPUT, "stream": "oper", "type": "gwt"},
                {
                    **BASE_OUTPUT,
                    "stream": "oper",
                    "type": "pfc",
                    "quantile": [f"{x}:{100}" for x in range(1, 101)],
                },
            ],
        ],
    ],
)
def test_ecpoint_outputs(inputs, perc_metadata, expected):
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
            }
        },
        inputs={"fc": {"source": {"type": "fdb"}}},
        outputs={
            "default": {"target": {"type": "fdb"}},
            "bs": {"metadata": {"type": "gbf"}},
            "wt": {"metadata": {"type": "gwt"}},
            "perc": {"metadata": {"type": "pfc", **perc_metadata}},
        },
        quantiles=[x / 100 for x in range(1, 101)],
    )
    config.print()
    outputs = list(config.out_mars(targets=["fdb"]))
    assert outputs == expected
