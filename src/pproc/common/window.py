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


def window_operation_from_config(window_config: dict) -> Dict[str, list]:
    """
    Derives window operation from config. If no window operation is explicitly
    specified then attempts to derive it from the thresholds.

    :param window_config: window configuration dictionary
    :return: dict mapping window operations to associated thresholds, if any
    """
    # Get window operation, or if not provided in config, derive from threshold
    window_operations = {}
    if "operation" in window_config:
        thresholds = window_config.get("thresholds", [])
        for threshold in thresholds:
            if isinstance(threshold["value"], str):
                threshold["value"] = float(threshold["value"])
        window_operations[window_config["operation"]] = thresholds
    elif "thresholds" in window_config:
        # Derive from threshold comparison parameter
        for threshold in window_config["thresholds"]:
            if isinstance(threshold["value"], str):
                threshold["value"] = float(threshold["value"])
            comparison = threshold["comparison"]
            if "<" in comparison:
                operation = "minimum"
            elif ">" in comparison:
                operation = "maximum"
            else:
                raise RuntimeError(f"Unknown threshold comparison {comparison}")
            window_operations.setdefault(operation, []).append(threshold)
    else:
        window_operations["aggregation"] = []

    return window_operations


def translate_window_config(
    coords: Union[list[Any], dict],
    window_operation: str,
    include_start: bool,
    grib_keys: Optional[dict] = None,
    deaccumulate: bool = False,
    **window_options,
) -> Tuple[str, dict]:
    """
    Create window configuration for the given operation

    :param coords: step range specification
    :param window operation: operation supported by accumulation
    :param grib_keys: additional grib keys to tie to the window
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

    operation = None
    extra = {}
    operation = window_operation
    if operation in [
        "sum",
        "minimum",
        "maximum",
        "mean",
        "aggregation",
        "standard_deviation",
    ]:
        if not include_init:
            coords = coords[1:]
    elif operation == "difference_rate":
        extra["factor"] = window_options.get("factor", 1.0)

    grib_header = {} if grib_keys is None else grib_keys.copy()
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
        "operation": operation,
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
        window_operations = window_operation_from_config(window_config)
        for operation, thresholds in window_operations.items():
            coords = window_config["coords"]
            for coord in coords:
                include_start = bool(window_config.get("include_start", False))
                acc_grib_keys = grib_keys.copy()
                acc_grib_keys.update(window_config.get("metadata", {}))
                window_name, acc_config = translate_window_config(
                    coord,
                    operation,
                    include_start,
                    acc_grib_keys,
                    window_config.get("deaccumulate", False),
                    factor=float(window_config.get("factor", 1.0)),
                )
                window_id = (
                    f"{prefix}{window_name}_{operation}_{window_index}"
                    if len(window_operations) > 1
                    else f"{prefix}{window_name}_{window_index}"
                )
                if thresholds:
                    acc_config["thresholds"] = thresholds
                yield window_id, acc_config


def legacy_window_factory(config: dict, grib_keys: dict) -> Iterator[Tuple[str, dict]]:
    yield from _iter_legacy_windows(config["windows"], grib_keys)
    yield from _iter_legacy_windows(
        config.get("std_anomaly_windows", []), grib_keys, prefix="std_"
    )
