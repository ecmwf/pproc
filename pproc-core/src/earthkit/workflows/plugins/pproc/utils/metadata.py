# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any


def window(operation: str, coords: list[Any], include_init: bool) -> dict:
    if len(coords) == 1:
        return {}

    ret = {}
    if operation == "diff":
        ret.update({"timeRangeIndicator": 5, "stepType": "diff"})
    if operation == "mean":
        ret["timeRangeIndicator"] = 3
        ret["numberIncludedInAverage"] = (
            len(coords) if include_init else len(coords) - 1
        )
        ret["numberMissingFromAveragesOrAccumulations"] = 0
    if operation in ["min", "max"]:
        ret["timeRangeIndicator"] = 2
    ret.setdefault("stepType", "max")
    ret["stepRange"] = f"{coords[0]}-{coords[-1]}"
    return ret
    