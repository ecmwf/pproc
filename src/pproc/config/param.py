from typing import Any, Dict, Optional
from typing_extensions import Self

import numpy as np
from pydantic import BaseModel, Field

from pproc.config.preprocessing import PreprocessingConfig


class ParamConfig(BaseModel):
    name: str
    sources: dict = {}
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    accumulations: dict = {}
    dtype_: str = Field(alias="dtype", default="float32")
    metadata: Dict[str, Any] = {}

    @property
    def dtype(self) -> type[Any]:
        return np.dtype(self.dtype_).type

    def in_keys(
        self, name: str, base: Optional[Dict[str, Any]] = None, **kwargs
    ) -> dict:
        keys = base.copy() if base is not None else {}
        keys.update(self.sources.get(name, {}).get("request", {}))
        keys.update(kwargs)
        return keys

    def merge(self, other: Self) -> Self:
        """
        Merge two parameter configurations different on different legacy window step accumulations
        """
        current = self.model_dump(by_alias=True)
        current_windows = (
            current.get("accumulations", {}).get("step", {}).pop("windows", None)
        )
        other = other.model_dump(by_alias=True)
        other_windows = (
            other.get("accumulations", {}).get("step", {}).pop("windows", None)
        )

        if current_windows is None or other_windows is None:
            raise ValueError(
                "Merging of two parameter configs step accumulations to be of type legacywindow"
            )

        if current != other:
            raise ValueError(
                "Merging of two parameter configs requires them to be the same except for window configurations"
            )

        # Merge different types of window configurations
        current["accumulations"]["step"]["windows"] = current_windows + [
            x for x in other_windows if x not in current_windows
        ]
        return type(self)(**current)
