# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from unittest.mock import patch
import pickle

import pytest
import yaml
from conflator import Conflator

from pproc.config import io


@pytest.mark.parametrize(
    "config, cli, expected",
    [
        [
            {"fc": {"request": {"class": "ai", "type": ["cf", "pf"]}}},
            ["--set", "fc=temp.grib"],
            {"fc": {"type": "file", "path": "temp.grib"}},
        ],
        [
            {"fc": {"request": {"class": "ai", "type": ["cf", "pf"]}}},
            ["--set", "fc=temp.grib", "--override-input", "class=od"],
            {
                "fc": {"type": "file", "path": "temp.grib"},
                "overrides": {"class": "od"},
            },
        ],
        [
            {
                "fc": {"request": {"class": "ai", "type": ["cf", "pf"]}},
                "default": {"type": "fdb"},
            },
            [],
            {"fc": {"type": "fdb", "request": {"class": "ai", "type": ["cf", "pf"]}}},
        ],
        [
            {
                "fc": {"request": {"class": "ai", "type": ["cf", "pf"]}},
                "default": {"type": "fdb"},
            },
            ["--in-fc", "fileset:temp.grib"],
            {"fc": {"type": "fileset", "path": "temp.grib"}},
        ],
    ],
    ids=["with-set", "with-overrides", "with-default", "with-source-override"],
)
def test_sources(tmpdir, config, cli, expected):
    with open(f"{tmpdir}/config.yaml", "w") as file:
        file.write(yaml.dump(config))
    source_model = io.create_source_model("test", ["fc"])
    with patch("sys.argv", ["", "-f", f"{tmpdir}/config.yaml"] + cli):
        cfg = Conflator(app_name="sources", model=source_model).load()
        assert cfg.model_dump(by_alias=True, exclude_defaults=True) == expected


@pytest.mark.parametrize(
    "config, cli, expected, target_type",
    [
        [
            {
                "default": {"target": {"type": "fdb"}, "metadata": {"class": "od"}},
            },
            [],
            {
                "default": {"target": {"type": "fdb"}, "metadata": {"class": "od"}},
                "test": {"target": {"type": "fdb"}, "metadata": {"class": "od"}},
                "overrides": {},
            },
            io.FDBTarget,
        ],
        [
            {
                "default": {"target": {"type": "fdb"}, "metadata": {"class": "od"}},
                "test": {"metadata": {"class": "ai", "type": "x"}},
            },
            [],
            {
                "default": {"target": {"type": "fdb"}, "metadata": {"class": "od"}},
                "test": {
                    "target": {"type": "fdb"},
                    "metadata": {"class": "ai", "type": "x"},
                },
                "overrides": {},
            },
            io.FDBTarget,
        ],
        [
            {"test": {"target": {"type": "fdb"}, "metadata": {"class": "od"}}},
            [
                "--set",
                "test.target.type=fileset",
                "--set",
                "test.target.path=temp.grib",
                "--override-output",
                "class=ai",
                "--override-output",
                "type=x",
            ],
            {
                "test": {
                    "target": {
                        "wrapped": {"type": "fileset", "path": "temp.grib"},
                        "overrides": {"class": "ai", "type": "x"},
                    },
                    "metadata": {"class": "od"},
                },
                "default": {"target": {"type": "null"}, "metadata": {}},
                "overrides": {"class": "ai", "type": "x"},
            },
            io.OverrideTargetWrapper,
        ],
        [
            {
                "default": {"target": {"type": "fdb"}, "metadata": {"class": "od"}},
            },
            ["--out-test", "fileset:temp.grib"],
            {
                "default": {"target": {"type": "fdb"}, "metadata": {"class": "od"}},
                "test": {
                    "target": {"type": "fileset", "path": "temp.grib"},
                    "metadata": {"class": "od"},
                },
                "overrides": {},
            },
            io.FileSetTarget,
        ],
    ],
    ids=[
        "config-defaults-only",
        "config-overrides",
        "with-set",
        "with-target-override",
    ],
)
def test_targets(tmpdir, config, cli, expected, target_type):
    with open(f"{tmpdir}/config.yaml", "w") as file:
        file.write(yaml.dump(config))
    targets_model = io.create_output_model("test", ["test"])
    with patch("sys.argv", ["", "-f", f"{tmpdir}/config.yaml"] + cli):
        cfg = Conflator(app_name="targets", model=targets_model).load()
        assert cfg.model_dump(by_alias=True) == expected
        assert type(cfg.test.target) == target_type


@pytest.mark.parametrize(
    "config, expected",
    [
        [
            {},
            {
                "default": {"target": {"type": "null"}, "metadata": {}},
                "test": {"target": {"type": "null"}, "metadata": {"type": "fcmean"}},
                "overrides": {},
            },
        ],
        [
            {"default": {"target": {"type": "fdb"}, "metadata": {"class": "od"}}},
            {
                "default": {"target": {"type": "fdb"}, "metadata": {"class": "od"}},
                "test": {
                    "target": {"type": "fdb"},
                    "metadata": {"class": "od", "type": "fcmean"},
                },
                "overrides": {},
            },
        ],
        [
            {"test": {"target": {"type": "fdb"}, "metadata": {"class": "od"}}},
            {
                "default": {"target": {"type": "null"}, "metadata": {}},
                "test": {
                    "target": {"type": "fdb"},
                    "metadata": {"class": "od", "type": "fcmean"},
                },
                "overrides": {},
            },
        ],
    ],
    ids=["no-config", "default-config", "test-config"],
)
def test_target_metadata(tmpdir, config, expected):
    with open(f"{tmpdir}/config.yaml", "w") as file:
        file.write(yaml.dump(config))
    targets_model = io.create_output_model("test", {"test": {"type": "fcmean"}})
    with patch("sys.argv", ["", "-f", f"{tmpdir}/config.yaml"]):
        cfg = Conflator(app_name="metadata", model=targets_model).load()
        assert cfg.model_dump(by_alias=True) == expected


def test_model_serialisation():
    output_models = [
        io.BaseSourceModel,
        io.BaseOutputModel,
        io.EnsmsOutputModel,
        io.AccumOutputModel,
        io.MonthlyStatsOutputModel,
        io.QuantilesOutputModel,
    ]
    for model in output_models:
        pickle.dumps(model)
