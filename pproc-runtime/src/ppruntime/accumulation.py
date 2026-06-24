from earthkit.data import FieldList

from ppruntime.utils import new_fieldlist


def difference_rate(*fields, factor: float = 1.0, metadata: dict) -> FieldList:
    values = fields[-1].values
    length = fields[-1].metadata()["step"]
    if len(fields) != 1:
        values = values - fields[0].values
        length = length - fields[0].metadata()["step"]
    return new_fieldlist(values / (factor * length), [fields[-1].metadata()], metadata)