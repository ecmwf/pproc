# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from pproc.config.param import ParamConfig

base_config = {
    "name": "2t",
    "accumulations": {
        "step": {
            "type": "legacywindow",
            "windows": [{"operation": "mean"}],
        }
    },
}


@pytest.mark.parametrize(
    "config2, merged",
    [
        [
            {
                "name": "2t",
                "accumulations": {
                    "step": {
                        "type": "legacywindow",
                        "windows": [
                            {"operation": "standard_deviation"},
                            {"operation": "mean", "include_start_step": True},
                        ],
                    }
                },
            },
            {
                "name": "2t",
                "accumulations": {
                    "step": {
                        "type": "legacywindow",
                        "windows": [
                            {"operation": "mean"},
                            {"operation": "standard_deviation"},
                            {"operation": "mean", "include_start_step": True},
                        ],
                    }
                },
            },
        ],
        [{**base_config, "name": "tp"}, None],
        [
            {
                "name": "2t",
                "accumulations": {
                    "step": {
                        "type": "legacywindow",
                        "windows": [{"operation": "standard_deviation"}],
                    },
                    "date": {},
                },
            },
            None,
        ],
        [base_config, base_config],
        [
            {
                "name": "2t",
                "accumulations": {
                    "step": {"operation": "mean", "coords": [[0, 6, 12]]},
                },
            },
            None,
        ],
    ],
    ids=["compatible", "diff_name", "diff_accum", "duplicate", "diff_window"],
)
def test_merge(config2, merged):
    param1 = ParamConfig(**base_config)
    param2 = ParamConfig(**config2)
    if merged is None:
        with pytest.raises(ValueError):
            param1.merge(param2)
    else:
        assert param1.merge(param2) == ParamConfig(**merged)
