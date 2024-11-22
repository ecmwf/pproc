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
