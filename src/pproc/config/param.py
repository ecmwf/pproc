from typing import Any, Dict, Iterator
from typing_extensions import Self
import itertools

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from pproc.config.preprocessing import PreprocessingConfig
from pproc.config.accumulation import AccumulationConfig
from pproc.config.utils import extract_mars
from pproc.config.io import Source, SourceCollection
from pproc.config.utils import update_request, expand


class ParamConfig(BaseModel):
    name: str
    sources: dict = {}
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    accumulations: dict[str, AccumulationConfig]
    dtype_: str = Field(alias="dtype", default="float32")
    metadata: Dict[str, Any] = {}

    @property
    def dtype(self) -> type[Any]:
        return np.dtype(self.dtype_).type

    def in_sources(
        self, sources: SourceCollection, name: str, **kwargs
    ) -> list[Source]:
        base_config: Source = getattr(sources, name)
        config = self.sources.get(name, {})
        reqs = update_request(
            base_config.request,
            config.get("request", {}),
            **kwargs,
            **sources.overrides,
        )
        if isinstance(reqs, dict):
            reqs = expand(reqs, "param")
        else:
            reqs = sum([list(expand(req, "param")) for req in reqs], [])

        return [
            Source(
                type=config.get("type", base_config.type),
                path=config.get("path", base_config.path),
                request=items.to_dict("records"),
            )
            for _, items in pd.DataFrame(reqs).groupby("param")
        ]

    def out_keys(self, sources: SourceCollection) -> Iterator:
        base_source: Source = getattr(sources, "fc")
        param_source = self.sources.get("fc", {})
        reqs = update_request(
            base_source.request,
            param_source.get("request", {}),
            **sources.overrides,
            **extract_mars(self.metadata),
        )
        if isinstance(reqs, dict):
            reqs = [reqs]
        for dim, accum in self.accumulations.items():
            reqs = [{**x, **y} for x, y in itertools.product(reqs, accum.out_mars(dim))]
        yield from reqs

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
