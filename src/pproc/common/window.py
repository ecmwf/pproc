# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Dict, Iterator, Optional, Tuple, Union, Any

from pproc.common.grib_helpers import fill_template_values


def translate_window_config(
    coords: Union[list[Any], dict],
    include_start: bool = False,
    metadata: Optional[dict] = None,
    deaccumulate: bool = False,
    **extra,
) -> Tuple[str, dict]:
    """
    Create window configuration for the given operation

    :param coords: step range specification
    :param include_start: if True, include first coord
    :param metadata: additional grib keys to tie to the window
    :param deaccumulate: if True, deaccumulate steps before performed window operation
    :return: Window name, Accumulation configuration dict
    :raises: ValueError for unsupported window operation string
    """
    if isinstance(coords, list):
        if len(coords) == 1 and isinstance(coords[0], str):
            start, end = list(map(int, coords[0].split("-")))
            include_start = True
        else:
            start = coords[0]
            end = coords[-1]
    else:
        start = coords.get("from", 0)
        end = coords["to"]
        by = coords.get("by", 1)
        coords = list(range(start, end + 1, by))

    name = str(end) if start == end else f"{start}-{end}"
    include_init = start == end or include_start
    if deaccumulate:
        if not include_init:
            raise ValueError("De-accumulation without `include_start` not allowed")
        if len(coords) < 2:
            raise ValueError("De-accumulation can not be performed on single coord")

    if not include_init:
        coords = coords[1:]

    grib_header = {} if metadata is None else metadata.copy()
    grib_header = fill_template_values(
        grib_header,
        {
            "num_coords": len(coords) - 1 * int(deaccumulate),
            "start_coord": start if not deaccumulate else coords[1],
            "end_coord": end,
        },
    )

    if end > start and end >= 256:
        if grib_header.get("edition", 1) == 1:
            # The range is encoded as two 8-bit integers
            grib_header.setdefault("unitOfTimeRange", 11)

    if start == end and "timeRangeIndicator" not in grib_header:
        if end >= 256:
            grib_header["timeRangeIndicator"] = 10
        elif end == 0:
            grib_header["timeRangeIndicator"] = 1
        else:
            grib_header["timeRangeIndicator"] = 0

    if start == end and "step" not in grib_header:
        grib_header["step"] = name
    else:
        grib_header.setdefault("stepType", "max")  # Don't override if set in config
        grib_header["stepRange"] = name

    acc_config = {
        "coords": coords,
        "sequential": True,
        "metadata": grib_header,
        "deaccumulate": deaccumulate,
        **extra,
    }

    return name, acc_config


def _iter_legacy_windows(
    windows: list,
    grib_keys: dict,
    prefix: str = "",
) -> Iterator[Tuple[str, dict]]:
    for window_index, window_config in enumerate(windows):
        window_config = window_config.copy()
        for coord in window_config.pop("coords"):
            coord_config = window_config.copy()
            acc_grib_keys = grib_keys.copy()
            acc_grib_keys.update(coord_config.pop("metadata", {}))
            window_name, acc_config = translate_window_config(
                coords=coord, metadata=acc_grib_keys, **coord_config
            )
            window_id = f"{prefix}{window_name}_{window_index}"
            yield window_id, acc_config


def legacy_window_factory(config: dict, grib_keys: dict) -> Iterator[Tuple[str, dict]]:
    yield from _iter_legacy_windows(config["windows"], grib_keys)
    yield from _iter_legacy_windows(
        config.get("std_anomaly_windows", []), grib_keys, prefix="std_"
    )
