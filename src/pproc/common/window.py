from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Set, Tuple, Union

from pproc.common.accumulation import Accumulation, Coord, create_accumulation
from pproc.common.steps import Step, step_to_coord
from pproc.common.stepseq import stepseq_monthly, stepseq_ranges


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
        window_operations["none"] = []

    return window_operations


def translate_window_config(
    window_options,
    window_operation: str,
    include_start: bool,
    grib_keys: Optional[dict] = None,
    deaccumulate: bool = False,
) -> Tuple[str, dict]:
    """
    Create window configuration for the given operation

    :param window_options: window range specification
    :param window operation: window operation: one of none, diff, add, minimum,
        maximum, weightedsum, diffdailyrate, mean, precomputed
    :param grib_keys: additional grib keys to tie to the window
    :param deaccumulate: if True, deaccumulate steps before performed window operation
    :return: Window name, Accumulation configuration dict
    :raises: ValueError for unsupported window operation string
    """
    if isinstance(window_options, list):
        start = window_options[0]
        end = window_options[-1]
        coords = window_options
    else:
        start = window_options.get("from", 0)
        end = window_options["to"]
        by = window_options.get("by", 1)
        coords = list(range(start, end + 1, by))

    size = end - start
    name = str(end) if size == 0 else f"{start}-{end}"
    include_init = size == 0 or include_start

    operation = None
    extra = {}
    if window_operation == "none":
        if not include_init:
            coords = coords[1:]
        if len(coords) > 1:
            raise ValueError(
                "Window operation can not be none for windows containing more than a single step"
            )
        operation = "aggregation"
    elif window_operation == "difference":
        operation = "difference"
    elif window_operation in ["sum", "minimum", "maximum", "mean", "aggregation"]:
        if not include_init:
            coords = coords[1:]
        operation = window_operation
    elif window_operation == "weightedsum":
        operation = "weighted_mean"
    elif window_operation == "diffdailyrate":
        extra["factor"] = 1.0 / 24.0
        operation = "difference_rate"
    elif window_operation == "difference_rate":
        extra["factor"] = window_options.get("factor", 1.0)
        operation = "difference_rate"
        operation = "aggregation"
    elif window_operation == "standard_deviation":
        if not include_init:
            coords = coords[1:]
        operation = "standard_deviation"

    if operation is None:
        raise ValueError(
            f"Unsupported window operation {window_operation}. Supported types: "
            + "difference, minimum, maximum, sun, weightedsum, diffdailyrate, mean, "
            + "aggregation and precomputed"
        )

    grib_header = {} if grib_keys is None else grib_keys.copy()

    if end > start and end >= 256:
        if grib_header.get("edition", 1) == 1:
            # The range is encoded as two 8-bit integers
            grib_header.setdefault("unitOfTimeRange", 11)

    if end == start and "timeRangeIndicator" not in grib_header:
        if end >= 256:
            grib_header["timeRangeIndicator"] = 10
        elif end == 0:
            grib_header["timeRangeIndicator"] = 1
        else:
            grib_header["timeRangeIndicator"] = 0

    if size == 0:
        grib_header["step"] = name
    else:
        grib_header.setdefault("stepType", "max")  # Don't override if set in config
        grib_header["stepRange"] = name

    acc_config = {
        "operation": operation,
        "coords": coords,
        "sequential": True,
        "grib_keys": grib_header,
        "deaccumulate": deaccumulate,
        **extra,
    }

    return name, acc_config


def create_window(
    window_options,
    window_operation: str,
    include_start: bool,
    grib_keys: Optional[dict] = None,
    deaccumulate: bool = False,
    return_name: bool = False,
) -> Union[Accumulation, Tuple[Accumulation, str]]:
    """
    Create window for the given operation

    :param window_options: window range specification
    :param window operation: window operation: one of none, diff, add, minimum,
        maximum, weightedsum, diffdailyrate, mean, precomputed
    :param grib_keys: additional grib keys to tie to the window
    :param deaccumulate: if True, deaccumulate steps before performed window operation
    :param return_name: if True, return the window name as well
    :return: Window instance that performs the operation, window name (only if
        `return_name` is True)
    :raises: ValueError for unsupported window operation string
    """
    name, config = translate_window_config(
        window_options, window_operation, include_start, grib_keys, deaccumulate
    )
    acc = create_accumulation(config)
    if return_name:
        return acc, name
    return acc


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
                acc_grib_keys.update(window_config.get("grib_keys", {}))
                window_name, acc_config = translate_window_config(
                    coord,
                    operation,
                    include_start,
                    acc_grib_keys,
                    window_config.get("deaccumulate", False),
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
