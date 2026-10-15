from typing import Any

from pproc.common.stepseq import fcmonth_to_steprange, timespan_to_steprange


def default_filter(request: dict, key: str) -> Any:
    if key not in request:
        raise ValueError(f"Filter key '{key}' not found in request {request}")
    return request[key]


def _steptype(request: dict, key: str) -> str:
    step = request.get("step", "")
    steprange = str(step).split("-")
    return "range" if len(steprange) == 2 else "instantaneous"


def _steplength(request: dict, key: str) -> str:
    if "fcmonth" in request:
        step = fcmonth_to_steprange(str(request["date"]), int(request["fcmonth"]))
    elif "timespan" in request:
        step = timespan_to_steprange(request["step"], request["timespan"])
    else:
        step = request.get("step", "")
    steprange = list(map(int, str(step).split("-")))
    length = str(0) if len(steprange) == 1 else str(steprange[1] - steprange[0])
    return length


def _selection(request: dict, key: str) -> str:
    return request.get("selection", "default")


def _members(request: dict, key: str) -> str:
    number = request["number"]
    if isinstance(number, (int, str)):
        number = [number]
    number = list(map(int, number))
    if 0 not in number:
        return "no_zero"
    if len(number) == 1:
        return "only_zero"
    return "contains_zero"
