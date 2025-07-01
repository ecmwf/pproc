def _steptype(request: dict, key: str) -> str:
    step = request.get("step", "")
    steprange = str(step).split("-")
    return "range" if len(steprange) == 2 else "instantaneous"


def _steplength(request: dict, key: str) -> str:
    step = request.get("step", "")
    steprange = list(map(int, str(step).split("-")))
    length = str(0) if len(steprange) == 1 else str(steprange[1] - steprange[0])
    return length


def _selection(request: dict, key: str) -> str:
    return request.get("selection", "default")
