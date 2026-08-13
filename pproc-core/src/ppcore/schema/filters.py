# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from ppcore.schema.exceptions import PProcSchemaError
from ppcore.utils.stepseq import fcmonth_to_steprange


def default_filter(request: dict, key: str) -> Any:
    if key not in request:
        raise PProcSchemaError(f"Filter key '{key}' not found in request {request}")
    return request[key]


def _steptype(request: dict, key: str) -> str:
    step = request.get("step", "")
    steprange = str(step).split("-")
    return "range" if len(steprange) == 2 else "instantaneous"


def _steplength(request: dict, key: str) -> str:
    if "fcmonth" in request:
        step = fcmonth_to_steprange(str(request["date"]), int(request["fcmonth"]))
    else:
        step = request.get("step", "")
    steprange = list(map(int, str(step).split("-")))
    length = str(0) if len(steprange) == 1 else str(steprange[1] - steprange[0])
    return length


def _selection(request: dict, key: str) -> str:
    return request.get("selection", "default")


def _members(request: dict, key: str) -> str:
    number = default_filter(request, "number")
    if isinstance(number, (int, str)):
        number = [number]
    number = list(map(int, number))
    if 0 not in number:
        return "no_zero"
    if len(number) == 1:
        return "only_zero"
    return "contains_zero"
