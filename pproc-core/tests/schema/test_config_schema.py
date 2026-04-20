# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from ppcore.schema.config import ConfigSchema

from conftest import schema


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


@pytest.mark.parametrize(
    "out_request, expected_config",
    [
        [
            {
                "stream": "enfo",
                "date": "20250101",
                "time": "00",
                "type": "sot",
                "number": 90,
                "param": "132167",
                "step": "0-24",
            },
            {"sot": [90]},
        ],
        [
            {
                "stream": "enfo",
                "date": "20250101",
                "time": "00",
                "type": "sot",
                "number": [10, 90],
                "param": "132167",
                "step": "0-24",
            },
            {"sot": [10, 90]},
        ],
        [
            {
                "stream": "enfo",
                "date": "20250101",
                "time": "00",
                "type": "pb",
                "quantile": "1:100",
                "param": "167",
                "step": "0-24",
            },
            {"quantiles": [0.01]},
        ],
        [
            {
                "stream": "enfo",
                "date": "20250101",
                "time": "00",
                "type": "pb",
                "quantile": ["1:100", "2:100"],
                "param": "167",
                "step": "0-24",
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
