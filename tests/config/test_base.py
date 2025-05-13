# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from unittest.mock import patch
import copy

import pytest
import yaml
from conflator import Conflator

from pproc.config.base import BaseConfig
from pproc.config.utils import deep_update


@pytest.fixture(scope="function")
def config(tmpdir) -> str:
    required = {
        "sources": {"fc": {"type": "fdb", "request": {}}},
        "parameters": [],
    }
    with open(f"{tmpdir}/config.yaml", "w") as file:
        file.write(yaml.dump(required))
    return f"{tmpdir}/config.yaml"


base_config = {
    "sources": {"fc": {"type": "fdb", "request": {}}},
    "parameters": {
        "param": {
            "sources": {"fc": {"request": {"param": "param"}}},
            "accumulations": {
                "step": {
                    "type": "legacywindow",
                    "windows": [{"operation": "mean"}],
                }
            },
        }
    },
}


@pytest.mark.parametrize(
    "cli_args, attr, expected",
    [
        [
            ["--override-input", "class=ai", "--override-input", "type=x"],
            "sources.overrides",
            {"class": "ai", "type": "x"},
        ],
        [
            ["--override-output", "class=ai", "--override-output", "type=x"],
            "outputs.overrides",
            {"class": "ai", "type": "x"},
        ],
        [["--log", "ERROR"], "log.level", "ERROR"],
        [["--recover"], "recovery.from_checkpoint", True],
    ],
    ids=["override-input", "override-output", "log-level", "recovery"],
)
def test_cli_overrides(config, cli_args, attr, expected):
    with patch("sys.argv", ["", "-f", config] + cli_args):
        cfg = Conflator(app_name="test", model=BaseConfig).load()
        field, cli = attr.split(".")
        assert getattr(getattr(cfg, field), cli) == expected


@pytest.mark.parametrize(
    "overrides, checkpointing, from_checkpoint",
    [
        [{}, True, False],
        [{"recovery": {"from_checkpoint": True}}, True, True],
        [{"recovery": {"enable_checkpointing": False}}, False, False],
    ],
)
def test_recovery(config, overrides, checkpointing, from_checkpoint):
    if len(overrides) > 0:
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)
            cfg.update(overrides)
        with open(config, "w") as f:
            f.write(yaml.dump(cfg))
    with patch("sys.argv", ["", "-f", config]):
        cfg = Conflator(app_name="test", model=BaseConfig).load()
    assert cfg.recovery.enable_checkpointing == checkpointing
    assert cfg.recovery.from_checkpoint == from_checkpoint


@pytest.mark.parametrize(
    "other, merged",
    [
        [
            {
                "sources": {"fc": {"type": "fdb", "request": {}}},
                "parameters": {
                    "tp": {
                        "sources": {"fc": {"request": {"param": "tp"}}},
                        "accumulations": {
                            "step": {
                                "type": "legacywindow",
                                "windows": [{"operation": "diff"}],
                            }
                        },
                    }
                },
            },
            {
                "sources": {"fc": {"type": "fdb", "request": {}}},
                "parameters": {
                    "param": {
                        "sources": {"fc": {"request": {"param": "param"}}},
                        "accumulations": {
                            "step": {
                                "type": "legacywindow",
                                "windows": [{"operation": "mean"}],
                            }
                        },
                    },
                    "tp": {
                        "sources": {"fc": {"request": {"param": "tp"}}},
                        "accumulations": {
                            "step": {
                                "type": "legacywindow",
                                "windows": [{"operation": "diff"}],
                            }
                        },
                    },
                },
            },
        ],
        [
            {
                "sources": {"fc": {"type": "fdb", "request": {}}},
                "parameters": {
                    "param": {
                        "sources": {"fc": {"request": {"param": "param"}}},
                        "accumulations": {
                            "step": {
                                "type": "legacywindow",
                                "windows": [{"operation": "standard_deviation"}],
                            }
                        },
                    }
                },
            },
            {
                "sources": {"fc": {"type": "fdb", "request": {}}},
                "parameters": {
                    "param": {
                        "sources": {"fc": {"request": {"param": "param"}}},
                        "accumulations": {
                            "step": {
                                "type": "legacywindow",
                                "windows": [
                                    {"operation": "mean"},
                                    {"operation": "standard_deviation"},
                                ],
                            }
                        },
                    },
                },
            },
        ],
        [{**base_config, "total_fields": 10}, None],
        [base_config, base_config],
    ],
    ids=["compat_diff_params", "compat_diff_windows", "diff_base", "duplicate"],
)
def test_merge(other: dict, merged: dict):
    config1 = BaseConfig(**base_config)
    config2 = BaseConfig(**other)
    if merged is None:
        with pytest.raises(ValueError):
            config1.merge(config2)
    else:
        assert config1.merge(config2) == BaseConfig(**merged)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        [
            {
                "sources": {
                    "fc": {
                        "request": [
                            {
                                "class": "od",
                                "stream": "enfo",
                                "type": "cf",
                            },
                            {
                                "class": "od",
                                "stream": "enfo",
                                "type": "pf",
                                "number": [1, 2, 3, 4, 5],
                            },
                        ],
                    }
                },
                "parameters": {
                    "param": {
                        "sources": {
                            "fc": {"request": {"param": "167", "levtype": "sfc"}}
                        },
                        "accumulations": {
                            "step": {
                                "type": "legacywindow",
                                "windows": [
                                    {
                                        "operation": "mean",
                                        "coords": [[x for x in range(0, 25, 6)]],
                                    }
                                ],
                            }
                        },
                    }
                },
            },
            [
                {
                    "class": "od",
                    "stream": "enfo",
                    "param": "167",
                    "levtype": "sfc",
                    "step": list(range(0, 25, 6)),
                    "type": "cf",
                    "source": "fdb",
                },
                {
                    "class": "od",
                    "stream": "enfo",
                    "param": "167",
                    "levtype": "sfc",
                    "step": list(range(0, 25, 6)),
                    "type": "pf",
                    "number": [1, 2, 3, 4, 5],
                    "source": "fdb",
                },
            ],
        ],
        [
            {
                "sources": {"fc": {"request": {"class": "od", "stream": "enfo"}}},
                "parameters": {
                    "param": {
                        "sources": {
                            "fc": {
                                "request": {
                                    "class": "ai",
                                    "param": "130",
                                    "levtype": "pl",
                                },
                                "type": "fileset",
                                "path": "data.grib",
                            }
                        },
                        "accumulations": {
                            "step": {
                                "type": "default",
                                "coords": [["0-24"], ["48-72"]],
                            },
                            "levelist": {"coords": [[250], [500]]},
                        },
                    }
                },
            },
            [
                {
                    "class": "ai",
                    "stream": "enfo",
                    "param": "130",
                    "levtype": "pl",
                    "levelist": [250, 500],
                    "step": ["0-24", "48-72"],
                    "source": "data.grib",
                }
            ],
        ],
        [
            {
                "sources": {
                    "fc": {"request": {"class": "od", "stream": "enfo"}},
                    "overrides": {"class": "ai"},
                },
                "parameters": {
                    "param": {
                        "sources": {
                            "fc": {
                                "request": {
                                    "param": "167",
                                    "levtype": "sfc",
                                },
                            }
                        },
                        "accumulations": {
                            "step": {
                                "type": "default",
                                "coords": [["0-24"], ["48-72"]],
                            },
                        },
                    }
                },
            },
            [
                {
                    "class": "ai",
                    "stream": "enfo",
                    "param": "167",
                    "levtype": "sfc",
                    "step": ["0-24", "48-72"],
                    "source": "fdb",
                }
            ],
        ],
        [
            {
                "parameters": {
                    "param": {
                        "preprocessing": [{"operation": "norm"}],
                        "sources": {
                            "fc": {
                                "request": [
                                    {
                                        "param": ["165", "166"],
                                        "class": "od",
                                        "stream": "enfo",
                                        "type": "cf",
                                    },
                                    {
                                        "param": ["165", "166"],
                                        "class": "od",
                                        "stream": "enfo",
                                        "type": "pf",
                                        "number": [1, 2, 3, 4, 5],
                                    },
                                ]
                            }
                        },
                        "accumulations": {
                            "step": {
                                "type": "legacywindow",
                                "windows": [
                                    {
                                        "operation": "mean",
                                        "coords": [[x for x in range(0, 25, 6)]],
                                    }
                                ],
                            }
                        },
                    }
                },
            },
            sum(
                [
                    [
                        {
                            "class": "od",
                            "stream": "enfo",
                            "param": param,
                            "step": list(range(0, 25, 6)),
                            "type": "cf",
                            "source": "fdb",
                        },
                        {
                            "class": "od",
                            "stream": "enfo",
                            "param": param,
                            "step": list(range(0, 25, 6)),
                            "type": "pf",
                            "source": "fdb",
                            "number": [1, 2, 3, 4, 5],
                        },
                    ]
                    for param in ["165", "166"]
                ],
                [],
            ),
        ],
    ],
    ids=["ensemble", "multi-accum", "overrides", "wind"],
)
def test_inputs(overrides: dict, expected: list[dict]):
    config = BaseConfig(**deep_update(copy.deepcopy(base_config), overrides))
    assert list(config.in_mars()) == expected
