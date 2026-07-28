# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from pproc.common.grib_helpers import fill_template_value


@pytest.mark.parametrize(
    "value, template_map, expected",
    [
        ("{start_coord}", {"start_coord": "20240507"}, "20240507"),
        ("{coords_span}:int", {"coords_span": 24}, 24),
        ("{start_coord[0:6]}", {"start_coord": "20240507"}, "202405"),
        ("{int(start_coord[0:6])}", {"start_coord": "20240507"}, 202405),
        ("{int(start_coord[0:4]) - 20}", {"start_coord": "20240507"}, 2004),
        (
            "clim_{start_coord[0:4]}",
            {"start_coord": "20240507"},
            "clim_{start_coord[0:4]}",
        ),
    ],
    ids=["no-type", "typed-int", "slice-str", "slice-int", "slice-expression", "noop"],
)
def test_fill_template_value(value, template_map, expected):
    assert fill_template_value(value, template_map) == expected
