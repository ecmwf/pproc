# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


def create_steps(step_config: list[dict]) -> list[int]:
    steps = set(
        sum(
            [list(range(x["from"], x["to"] + 1, x.get("by", 1))) for x in step_config],
            [],
        )
    )
    return sorted(steps)
