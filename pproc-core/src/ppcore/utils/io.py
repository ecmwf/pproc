from typing import Optional, Tuple
import re

_LOCATION_RE = re.compile("^([a-z](?:[a-z0-9+-.])*):(.*)$", re.I)


def split_location(
    loc: str, default: Optional[str] = None
) -> Tuple[Optional[str], str]:
    m = _LOCATION_RE.fullmatch(loc)
    if m is None:
        return (default, loc)
    return m.groups()
