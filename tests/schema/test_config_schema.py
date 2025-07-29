# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pproc.schema.config import ConfigSchema

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
