import pytest

from pproc.config import types
from pproc.config.utils import deep_update

BASE_OUTPUT = {
    "target": "fdb",
}


@pytest.mark.parametrize(
    "overrides, expected",
    [
        [
            {
                "parameters": {
                    "2t": {
                        "accumulations": {
                            "step": {
                                "type": "legacywindow",
                                "windows": [
                                    {
                                        "operation": "mean",
                                        "name": {"length": 24},
                                        "coords": [{"from": 18, "to": 36, "by": 6}],
                                    },
                                    {
                                        "operation": "mean",
                                        "name": {"length": 72},
                                        "coords": [{"from": 18, "to": 84, "by": 6}],
                                    },
                                ],
                            },
                        }
                    }
                },
            },
            [
                {**BASE_OUTPUT, "stream": "enfo", "type": "efi", "step": ["12-36"]},
                {**BASE_OUTPUT, "stream": "enfo", "type": "efic", "step": ["12-36"]},
                {**BASE_OUTPUT, "stream": "enfo", "type": "efi", "step": ["12-84"]},
                {**BASE_OUTPUT, "stream": "enfo", "type": "efic", "step": ["12-84"]},
                {
                    **BASE_OUTPUT,
                    "stream": "enfo",
                    "type": "sot",
                    "step": ["12-36"],
                    "number": [10, 90],
                },
                {
                    **BASE_OUTPUT,
                    "stream": "enfo",
                    "type": "sot",
                    "step": ["12-84"],
                    "number": [10, 90],
                },
            ],
        ],
        [
            {
                "parameters": {"2t": {"sot": []}},
            },
            [
                {**BASE_OUTPUT, "stream": "enfo", "type": "efi", "step": ["12-36"]},
                {**BASE_OUTPUT, "stream": "enfo", "type": "efic", "step": ["12-36"]},
            ],
        ],
        [
            {
                "outputs": {"efi": {"target": "file", "path": "efi.grib"}},
            },
            [
                {
                    **BASE_OUTPUT,
                    "stream": "enfo",
                    "type": "sot",
                    "step": ["12-36"],
                    "number": [10, 90],
                },
            ],
        ],
    ],
    ids=["mult-accum", "no-sot", "diff-target"],
)
def test_outputs(overrides, expected):
    base_config = {
        "parameters": {
            "2t": {
                "inputs": {
                    "fc": {
                        "request": [
                            {"stream": "oper", "type": "fc"},
                            {"stream": "enfo", "type": "pf", "number": [1, 2, 3]},
                        ]
                    },
                    "clim": {"request": {"type": "cd"}},
                },
                "sot": [10, 90],
                "accumulations": {
                    "step": {
                        "operation": "mean",
                        "name": {"length": 24},
                        "coords": [{"from": 18, "to": 36, "by": 6}],
                    },
                },
            }
        },
        "inputs": {"default": {"source": {"type": "fdb"}}},
        "outputs": {
            "default": {"target": {"type": "fdb"}, "metadata": {"stream": "enfo"}},
            "efi": {},
            "sot": {},
            "cpf": {},
        },
    }
    config = types.ExtremeConfig(**deep_update(base_config, overrides))
    config.print()
    outputs = list(config.out_mars(targets=["fdb"]))
    assert outputs == expected
