from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Set, Tuple, Union

from pproc.common.accumulation import Accumulation, Coord, create_accumulation
from pproc.common.steps import Step, step_to_coord


@dataclass
class WindowConfig:
    start: int
    end: int
    step: int
    name: str
    suffix: str
    steps: list
    include_init: bool


def parse_window_config(config: dict, include_init: bool = True) -> WindowConfig:
    start = int(config["range"][0])
    end = int(config["range"][1])
    step = int(config["range"][2]) if len(config["range"]) > 2 else 1
    window_size = end - start
    name = str(end) if window_size == 0 else f"{start}-{end}"
    suffix = f"{window_size:0>3}_{start:0>3}h_{end:0>3}h"
    if include_init:
        steps = list(range(start, end + 1, step))
    else:
        steps = list(range(start + step, end + 1, step))
    return WindowConfig(start, end, step, name, suffix, steps, include_init)


def window_operation_from_config(window_config: dict) -> Dict[str, list]:
    """
    Derives window operation from config. If no window operation is explicitly
    specified then attempts to derive it from the thresholds.

    :param window_config: window configuration dictionary
    :return: dict mapping window operations to associated thresholds, if any
    """
    # Get window operation, or if not provided in config, derive from threshold
    window_operations = {}
    if "window_operation" in window_config:
        thresholds = window_config.get("thresholds", [])
        for threshold in thresholds:
            if isinstance(threshold["value"], str):
                threshold["value"] = float(threshold["value"])
        window_operations[window_config["window_operation"]] = thresholds
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
) -> Tuple[str, dict]:
    """
    Create window configuration for the given operation

    :param window_options: window range specification
    :param window operation: window operation: one of none, diff, add, minimum,
        maximum, weightedsum, diffdailyrate, mean, precomputed
    :param grib_keys: additional grib keys to tie to the window
    :return: Window name, Accumulation configuration dict
    :raises: ValueError for unsupported window operation string
    """
    include_init = (
        window_options["range"][0] == window_options["range"][1]
    ) or include_start

    config = None
    operation = None
    coords = None
    extra = {}
    if window_operation == "none":
        config = parse_window_config(window_options, include_init)
        if len(config.steps) > 1:
            raise ValueError(
                "Window operation can not be none for windows containing more than a single step"
            )
        operation = "aggregation"
    elif window_operation == "diff":
        config = parse_window_config(window_options, True)
        config.steps = [config.start, config.end]
        operation = "difference"
    elif window_operation == "add":
        config = parse_window_config(window_options, include_init)
        operation = "sum"
    elif window_operation in ["minimum", "maximum", "mean", "aggregation"]:
        config = parse_window_config(window_options, include_init)
        operation = window_operation
    elif window_operation == "weightedsum":
        config = parse_window_config(window_options, False)
        operation = "weighted_mean"
        coords = [config.start] + config.steps
    elif window_operation == "diffdailyrate":
        config = parse_window_config(window_options, True)
        config.steps = [config.start, config.end]
        extra["factor"] = 1.0 / 24.0
        operation = "difference_rate"
    elif window_operation == "precomputed":
        config = parse_window_config(window_options, True)
        config.steps = [Step(config.start, config.end)]
        operation = "aggregation"
    elif window_operation == "standard_deviation":
        config = parse_window_config(window_options, include_init)
        operation = "standard_deviation"

    if config is None:
        raise ValueError(
            f"Unsupported window operation {window_operation}. Supported types: "
            + "diff, minimum, maximum, add, weightedsum, diffdailyrate, mean, "
            + "aggregation and precomputed"
        )

    if coords is None:
        coords = [step_to_coord(step) for step in config.steps]

    grib_header = {} if grib_keys is None else grib_keys.copy()

    if config.end > config.start and config.end >= 256:
        if grib_header.get("edition", 1) == 1:
            # The range is encoded as two 8-bit integers
            grib_header.setdefault("unitOfTimeRange", 11)

    if config.end == config.start and "timeRangeIndicator" not in grib_header:
        if config.end >= 256:
            grib_header["timeRangeIndicator"] = 10
        elif config.end == 0:
            grib_header["timeRangeIndicator"] = 1
        else:
            grib_header["timeRangeIndicator"] = 0

    if config.end == config.start:
        grib_header["step"] = config.name
    else:
        grib_header.setdefault("stepType", "max")  # Don't override if set in config
        grib_header["stepRange"] = config.name

    acc_config = {
        "operation": operation,
        "coords": coords,
        "sequential": True,
        "grib_keys": grib_header,
        **extra,
    }

    return config.name, acc_config


def create_window(
    window_options,
    window_operation: str,
    include_start: bool,
    grib_keys: Optional[dict] = None,
    return_name: bool = False,
) -> Union[Accumulation, Tuple[Accumulation, str]]:
    """
    Create window for the given operation

    :param window_options: window range specification
    :param window operation: window operation: one of none, diff, add, minimum,
        maximum, weightedsum, diffdailyrate, mean, precomputed
    :param grib_keys: additional grib keys to tie to the window
    :param return_name: if True, return the window name as well
    :return: Window instance that performs the operation, window name (only if
        `return_name` is True)
    :raises: ValueError for unsupported window operation string
    """
    name, config = translate_window_config(
        window_options, window_operation, include_start, grib_keys
    )
    acc = create_accumulation(config)
    if return_name:
        return acc, name
    return acc


def _iter_legacy_windows(
    windows: list,
    grib_keys: dict,
    coords_override: Optional[Set[Coord]] = None,
    prefix: str = "",
) -> Iterator[Tuple[str, dict]]:
    for window_index, window_config in enumerate(windows):
        window_operations = window_operation_from_config(window_config)
        for operation, thresholds in window_operations.items():
            for period in window_config["periods"]:
                include_start = bool(window_config.get("include_start_step", False))
                acc_grib_keys = grib_keys.copy()
                acc_grib_keys.update(window_config.get("grib_set", {}))
                window_name, acc_config = translate_window_config(
                    period, operation, include_start, acc_grib_keys
                )
                window_id = (
                    f"{prefix}{window_name}_{operation}_{window_index}"
                    if len(window_operations) > 1
                    else f"{prefix}{window_name}_{window_index}"
                )
                if coords_override is not None:
                    acc_config["coords"] = sorted(
                        coords_override.intersection(acc_config["coords"])
                    )
                if thresholds:
                    acc_config["thresholds"] = thresholds
                yield window_id, acc_config


def legacy_window_factory(config: dict, grib_keys: dict) -> Iterator[Tuple[str, dict]]:
    coords_override = None
    if "steps" in config:
        coords_override = set()
        for steps in config["steps"]:
            start_step = steps["start_step"]
            end_step = steps["end_step"]
            interval = steps["interval"]
            range_len = steps.get("range", None)

            if range_len is None:
                coords_override.update(range(start_step, end_step + 1, interval))
            else:
                for sstep in range(start_step, end_step - range_len + 1, interval):
                    coords_override.add(step_to_coord(Step(sstep, sstep + range_len)))

    yield from _iter_legacy_windows(config["windows"], grib_keys, coords_override)
    yield from _iter_legacy_windows(
        config.get("std_anomaly_windows", []), grib_keys, coords_override, prefix="std_"
    )
