from unittest.mock import patch

import pytest
import yaml
from conflator import Conflator

from pproc.config.base import BaseConfig


@pytest.fixture(scope="function")
def config(tmpdir) -> str:
    required = {
        "members": 5,
        "sources": {"default": {"type": "fdb", "request": {}}},
        "parameters": [],
    }
    with open(f"{tmpdir}/config.yaml", "w") as file:
        file.write(yaml.dump(required))
    return f"{tmpdir}/config.yaml"


base_config = {
    "members": 5,
    "sources": {"default": {"type": "fdb", "request": {}}},
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
                "sources": {"default": {"type": "fdb", "request": {}}},
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
                "sources": {"default": {"type": "fdb", "request": {}}},
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
                "sources": {"default": {"type": "fdb", "request": {}}},
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
                "sources": {"default": {"type": "fdb", "request": {}}},
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
    print("CONFIG1", config1)
    config2 = BaseConfig(**other)
    if merged is None:
        with pytest.raises(ValueError):
            config1.merge(config2)
    else:
        assert config1.merge(config2) == BaseConfig(**merged)
