# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import schema

from ppcore.schema.config import ConfigSchema


def test_reconstruct():
    config_schema = ConfigSchema(schema("config"))
    cfgs = list(config_schema.reconstruct(entrypoint="pproc-ensms"))
    assert len(cfgs) != 0
    for out, cfg in cfgs:
        assert cfg["entrypoint"] == "pproc-ensms"
        assert out["type"] in ["em", "es", "taem", "taes"]
    assert len([x for x, _ in cfgs if x["type"] == "em"]) == len(
        [x for x, _ in cfgs if x["type"] == "es"]
    )


def test_reconstruct_cache():
    config_schema = ConfigSchema(
        schema("config"),
        matching_cache_size=2,
    )
    cfgs_a = list(config_schema.reconstruct(entrypoint="pproc-ensms"))

    cfgs_b = list(config_schema.reconstruct(entrypoint="pproc-ensms"))
    assert cfgs_a == cfgs_b
    assert len(config_schema._matching_cache) == 1

    config_schema.clear_matching_cache()
    assert len(config_schema._matching_cache) == 0


def test_reconstruct_dfs():
    config_schema = ConfigSchema(schema("config"))
    dfs_cfgs = list(
        config_schema.reconstruct(
            entrypoint="pproc-ensms", method="dfs", enable_cache=False
        )
    )
    bfs_cfgs = list(
        config_schema.reconstruct(
            entrypoint="pproc-ensms", method="bfs", enable_cache=False
        )
    )
    assert dfs_cfgs[0] == bfs_cfgs[0]
    assert len(dfs_cfgs) == len(bfs_cfgs)


@pytest.mark.parametrize(
    "out_request, expected_config",
    [
        [
            {
                "class": "od",
                "stream": "enfo",
                "date": "20250101",
                "time": "00",
                "type": "sot",
                "number": 90,
                "param": "132167",
                "step": "0-24",
                "levtype": "sfc",
            },
            {"sot": [90]},
        ],
        [
            {
                "class": "od",
                "stream": "enfo",
                "date": "20250101",
                "time": "00",
                "type": "sot",
                "number": [10, 90],
                "param": "132167",
                "step": "0-24",
                "levtype": "sfc",
            },
            {"sot": [10, 90]},
        ],
        [
            {
                "class": "od",
                "stream": "enfo",
                "date": "20250101",
                "time": "00",
                "type": "pb",
                "quantile": "1:100",
                "param": "167",
                "step": "0-24",
                "levtype": "sfc",
            },
            {"quantiles": [0.01]},
        ],
        [
            {
                "class": "od",
                "stream": "enfo",
                "date": "20250101",
                "time": "00",
                "type": "pb",
                "quantile": ["1:100", "2:100"],
                "param": "167",
                "step": "0-24",
                "levtype": "sfc",
            },
            {"quantiles": [0.01, 0.02]},
        ],
    ],
)
def test_config_from_output(out_request, expected_config):
    config_schema = ConfigSchema(schema("config"))
    config = config_schema.config(out_request)
    for key, value in expected_config.items():
        assert config[key] == value
