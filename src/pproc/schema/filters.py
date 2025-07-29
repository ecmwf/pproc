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


def _number(request: dict, key: str) -> str:
    number = request["number"]
    if isinstance(number, int):
        number = [number]
    if 0 not in number:
        return "no_zero"
    if len(number) == 1:
        return "only_zero"
    return "contains_zero"
