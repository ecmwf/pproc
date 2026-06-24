import array_api_compat
from earthkit.data import FieldList

from ppruntime.utils import new_fieldlist


def difference_rate(*fields, factor: float = 1.0, metadata: dict = {}) -> FieldList:
    if len(fields) not in [1, 2]:
        raise ValueError("difference_rate expects 1 or 2 FieldList arguments")
    xp = array_api_compat.array_namespace(fields[-1].values)
    values = fields[-1].values
    length = xp.asarray(fields[-1].metadata("step"))
    if len(fields) != 1:
        values = values - fields[0].values
        length = length - xp.asarray(fields[0].metadata("step"))
    return new_fieldlist(values / (factor * length), fields[-1].metadata(), metadata)
