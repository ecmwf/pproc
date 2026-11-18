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
                "outputs": {
                    "default": {"metadata": {"stream": "enfo"}},
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
                "outputs": {
                    "default": {"metadata": {"stream": "enfo"}},
                },
            },
            [
                {**BASE_OUTPUT, "stream": "enfo", "type": "efi", "step": ["12-36"]},
                {**BASE_OUTPUT, "stream": "enfo", "type": "efic", "step": ["12-36"]},
            ],
        ],
        [
            {
                "outputs": {
                    "default": {"metadata": {"stream": "enfo"}},
                    "efi": {"target": "file", "path": "efi.grib"},
                },
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
        [
            {
                "parameters": {
                    "2t": {
                        "inputs": {
                            "fc": {
                                "request": [
                                    {"stream": "eefo", "type": "cf"},
                                    {
                                        "stream": "eefo",
                                        "type": "pf",
                                        "number": [1, 2, 3],
                                    },
                                ]
                            },
                        },
                        "accumulations": {
                            "step": {
                                "type": "legacywindow",
                                "windows": [
                                    {
                                        "coords": [["0-168"], ["24-192"]],
                                    },
                                ],
                            },
                        },
                    }
                },
            },
            [
                {
                    **BASE_OUTPUT,
                    "stream": "eefo",
                    "type": "efi",
                    "step": ["0-168", "24-192"],
                },
                {
                    **BASE_OUTPUT,
                    "stream": "eefo",
                    "type": "efic",
                    "step": ["0-168", "24-192"],
                },
                {
                    **BASE_OUTPUT,
                    "stream": "eefo",
                    "type": "sot",
                    "step": ["0-168", "24-192"],
                    "number": [10, 90],
                },
            ],
        ],
        [
            {
                "parameters": {
                    "2t": {
                        "inputs": {
                            "fc": {
                                "request": [
                                    {
                                        "stream": "oper",
                                        "type": "fc",
                                        "timespan": "none",
                                    },
                                    {
                                        "stream": "enfo",
                                        "type": "pf",
                                        "number": [1, 2, 3],
                                        "timespan": "none",
                                    },
                                ]
                            },
                            "clim": {"request": {"type": "cd"}},
                        },
                    },
                },
                "outputs": {
                    "default": {"metadata": {"stream": "enfo"}},
                },
            },
            [
                {
                    **BASE_OUTPUT,
                    "stream": "enfo",
                    "type": "efi",
                    "step": [36],
                    "timespan": "24h",
                },
                {
                    **BASE_OUTPUT,
                    "stream": "enfo",
                    "type": "efic",
                    "step": [36],
                    "timespan": "24h",
                },
                {
                    **BASE_OUTPUT,
                    "stream": "enfo",
                    "type": "sot",
                    "number": [10, 90],
                    "step": [36],
                    "timespan": "24h",
                },
            ],
        ],
    ],
    ids=["mult-accum", "no-sot", "diff-target", "subseasonal", "timespan"],
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
            "default": {"target": {"type": "fdb"}},
            "efi": {},
            "sot": {},
            "cpf": {},
        },
    }
    config = types.ExtremeConfig(**deep_update(base_config, overrides))
    config.print()
    outputs = list(config.out_mars(targets=["fdb"]))
    assert outputs == expected


def _extreme_param(sot: list[int], clim: dict = None, clim_step: dict = None):
    clim_request = {"type": "cd"}
    if clim_step is not None:
        clim_request["step"] = clim_step
    config = {
        "name": "2t",
        "inputs": {
            "fc": {"request": {"stream": "enfo", "type": "pf"}},
            "clim": {"request": clim_request},
        },
        "sot": sot,
    }
    if clim is not None:
        config["clim"] = clim
    return types.ExtremeParamConfig(**config)


@pytest.mark.parametrize(
    "clim1, clim2, clim_steps, expected",
    [
        [
            {"metadata": {"expver": "0001"}, "dtype": "float64"},
            {"metadata": {"expver": "0001"}, "dtype": "float64"},
            None,
            {"metadata": {"expver": "0001"}, "dtype_": "float64"},
        ],
        [
            {"metadata": {"expver": "0001"}},
            {"metadata": {"expver": "0002"}},
            None,
            None,
        ],
        [None, {"dtype": "float64"}, None, None],
        [
            {"metadata": {"expver": "0001"}},
            {"metadata": {"expver": "0001"}},
            ({"0-24": 12}, {"24-48": 36}),
            {"metadata": {"expver": "0001"}},
        ],
    ],
    ids=[
        "same-clim",
        "different-metadata",
        "different-dtype",
        "different-inputs",
    ],
)
def test_merge_clim(clim1, clim2, clim_steps, expected):
    step1, step2 = clim_steps or (None, None)
    param1 = _extreme_param([10], clim=clim1, clim_step=step1)
    param2 = _extreme_param([90], clim=clim2, clim_step=step2)

    if expected is None:
        with pytest.raises(ValueError, match="Cannot merge different clim configs"):
            param1.merge(param2)
        return

    merged = param1.merge(param2)
    assert merged.sot == [10, 90]
    for attr, value in expected.items():
        assert getattr(merged.clim, attr) == value
    assert merged.clim.model_dump(
        by_alias=True, exclude={"inputs"}
    ) == param1.clim.model_dump(by_alias=True, exclude={"inputs"})

    expected_clim_request = {"type": "cd"}
    if clim_steps is not None:
        expected_clim_request["step"] = {**step1, **step2}
    assert merged.inputs["clim"]["request"] == expected_clim_request
    assert merged.clim.inputs == merged.inputs
