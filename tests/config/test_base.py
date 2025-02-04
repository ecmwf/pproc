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
        "members": 5,
        "sources": {"fc": {"type": "fdb", "request": {}}},
        "parameters": [],
    }
    with open(f"{tmpdir}/config.yaml", "w") as file:
        file.write(yaml.dump(required))
    return f"{tmpdir}/config.yaml"


base_config = {
    "members": 5,
    "sources": {"fc": {"type": "fdb", "request": {}}},
    "parameters": {
        "2t": {
            "accumulations": {
                "step": {
                    "type": "legacywindow",
                    "windows": [{"window_operation": "mean"}],
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
    ],
    ids=["override-input", "override-output", "log-level"],
)
def test_cli_overrides(config, cli_args, attr, expected):
    with patch("sys.argv", ["", "-f", config] + cli_args):
        cfg = Conflator(app_name="test", model=BaseConfig).load()
        field, cli = attr.split(".")
        assert getattr(getattr(cfg, field), cli) == expected


@pytest.mark.parametrize(
    "other, merged",
    [
        [
            {
                "members": 5,
                "sources": {"fc": {"type": "fdb", "request": {}}},
                "parameters": {
                    "tp": {
                        "accumulations": {
                            "step": {
                                "type": "legacywindow",
                                "windows": [{"window_operation": "diff"}],
                            }
                        },
                    }
                },
            },
            {
                "members": 5,
                "sources": {"fc": {"type": "fdb", "request": {}}},
                "parameters": {
                    "2t": {
                        "accumulations": {
                            "step": {
                                "type": "legacywindow",
                                "windows": [{"window_operation": "mean"}],
                            }
                        },
                    },
                    "tp": {
                        "accumulations": {
                            "step": {
                                "type": "legacywindow",
                                "windows": [{"window_operation": "diff"}],
                            }
                        },
                    },
                },
            },
        ],
        [
            {
                "members": 5,
                "sources": {"fc": {"type": "fdb", "request": {}}},
                "parameters": {
                    "2t": {
                        "accumulations": {
                            "step": {
                                "type": "legacywindow",
                                "windows": [{"window_operation": "standard_deviation"}],
                            }
                        },
                    }
                },
            },
            {
                "members": 5,
                "sources": {"fc": {"type": "fdb", "request": {}}},
                "parameters": {
                    "2t": {
                        "accumulations": {
                            "step": {
                                "type": "legacywindow",
                                "windows": [
                                    {"window_operation": "mean"},
                                    {"window_operation": "standard_deviation"},
                                ],
                            }
                        },
                    },
                },
            },
        ],
        [{**base_config, "members": 10}, None],
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
                        "request": {
                            "class": "od",
                            "stream": "enfo",
                            "type": ["cf", "pf"],
                        }
                    }
                },
                "parameters": {
                    "2t": {
                        "sources": {
                            "fc": {"request": {"param": "167", "levtype": "sfc"}}
                        },
                        "accumulations": {
                            "step": {
                                "type": "legacywindow",
                                "windows": [
                                    {
                                        "window_operation": "mean",
                                        "periods": [{"range": [0, 24, 6]}],
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
                    "2t": {
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
                    "2t": {
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
    ],
    ids=["ensemble", "multi-accum", "overrides"],
)
def test_inputs(overrides: dict, expected: list[dict]):
    config = BaseConfig(**deep_update(copy.deepcopy(base_config), overrides))
    assert list(config.in_mars()) == expected
