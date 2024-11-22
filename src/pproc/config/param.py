from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel


class ParamConfig(BaseModel):
    name: str
    sources: dict = {}
    preprocessing: Optional[list] = []
    accumulations: dict = {}
    dtype: type = np.float32
    metadata: Dict[str, Any] = {}

    def in_keys(self, name: str, base: Optional[Dict[str, Any]] = None, **kwargs):
        keys = base.copy() if base is not None else {}
        keys.update(self.sources.get(name, {}).get("request", {}))
        keys.update(kwargs)
        return keys
