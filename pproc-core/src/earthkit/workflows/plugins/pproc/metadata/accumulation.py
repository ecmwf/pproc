# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from earthkit.workflows.plugins.pproc.config.accumulation import Coords
from earthkit.workflows.plugins.pproc.utils.metadata import fill_template_values


def accumulation_metadata(
    dim: str, coords: Coords, name: str, metadata: Optional[dict] = None
) -> dict:
    metadata = (metadata or {}).copy()
    if dim == "step":
        steps = name.split("-")
        if len(steps) > 1:
            start = int(steps[0].split("_")[-1])
            end = int(steps[-1].split("_")[0])
        else:
            start = coords[0]
            end = start

        if end != start:
            assert isinstance(start, int) and isinstance(end, int)
            assert end > start, "Start step can not be greater than end step"
            steprange = f"{start}-{end}"
            if end >= 256 and metadata.get("edition", 1) == 1:
                # The range is encoded as two 8-bit integers
                metadata.setdefault("unitOfTimeRange", 11)
            if coords[0] != steprange:
                metadata.setdefault(
                    "stepType", "max"
                )  # Don't override if set in config
            metadata["stepRange"] = steprange
        else:
            assert end == start
            if "timeRangeIndicator" not in metadata:
                assert isinstance(end, int)
                if end >= 256:
                    metadata["timeRangeIndicator"] = 10
                elif end == 0:
                    metadata["timeRangeIndicator"] = 1
                else:
                    metadata["timeRangeIndicator"] = 0
            metadata.setdefault("step", str(start))
    else:
        start = coords[0]
        end = coords[-1]
    return fill_template_values(
        metadata,
        {
            "num_coords": len(coords),
            "start_coord": start,
            "end_coord": end,
        },
    )
