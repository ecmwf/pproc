# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any, Dict, Iterator
from typing_extensions import Self
import itertools
import os
import logging
import copy

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from pproc.config.preprocessing import PreprocessingConfig
from pproc.config.accumulation import AccumulationConfig
from pproc.config.utils import extract_mars
from pproc.config.io import Source, SourceCollection
from pproc.config.utils import update_request, expand

logger = logging.getLogger(__name__)


class ParamConfig(BaseModel):
    name: str
    sources: dict = {}
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    accumulations: dict[str, AccumulationConfig] = Field(default_factory=dict)
    dtype_: str = Field(alias="dtype", default="float32")
    metadata: Dict[str, Any] = {}
    total_fields: int = 1
    vod2uv: bool = False

    @model_validator(mode="after")
    def set_vod2uv(self) -> Self:
        for src_config in self.sources.values():
            req = src_config.get("request", {})
            if isinstance(req, list):
                req = req[0]
            self.vod2uv = req.get("interpolate", {}).get("vod2uv", False)
            break
        if self.vod2uv:
            self.total_fields = 2
        return self

    @property
    def dtype(self) -> type[Any]:
        return np.dtype(self.dtype_).type

    def in_sources(
        self, sources: SourceCollection, name: str, **kwargs
    ) -> list[Source]:
        base_config: Source = getattr(sources, name)
        config = self.sources.get(name, {})
        reqs = update_request(
            copy.deepcopy(base_config.request),
            config.get("request", {}),
            **kwargs,
            **sources.overrides,
        )

        if self.vod2uv:
            return [
                Source(
                    type=config.get("type", base_config.type),
                    path=config.get("path", base_config.path),
                    request=reqs,
                )
            ]

        reqs = list(expand(reqs, "param"))
        df = pd.DataFrame(reqs)
        if "param" in df:
            return [
                Source(
                    type=config.get("type", base_config.type),
                    path=config.get("path", base_config.path),
                    request=[row.dropna().to_dict() for _, row in items.iterrows()],
                )
                for _, items in df.groupby("param")
            ]
        return [
            Source(
                type=config.get("type", base_config.type),
                path=config.get("path", base_config.path),
                request=reqs,
            )
        ]

    def out_keys(self, sources: SourceCollection) -> Iterator:
        fc_name = sources.names[0]
        base_source: Source = getattr(sources, fc_name)
        param_source = self.sources.get(fc_name, {})
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
        current = self.model_dump(by_alias=True, exclude=("accumulations",))

        if current != other.model_dump(by_alias=True, exclude=("accumulations",)):
            logger.debug(
                "Current: \n %s, other \n %s",
                current,
                other.model_dump(by_alias=True, exclude=("accumulations",)),
            )
            raise ValueError(
                "Merging of two parameter configs requires them to be the same, except for accumulations"
            )

        if self.accumulations.keys() != other.accumulations.keys():
            raise ValueError(
                "Merging of two parameter configs requires them to have the same accumulations dimensions"
            )
        current["accumulations"] = {}
        for dim in other.accumulations.keys():
            if dim == "step":
                current["accumulations"][dim] = (
                    self.accumulations[dim]
                    .merge(other.accumulations[dim])
                    .model_dump(by_alias=True)
                )
            else:
                if self.accumulations[dim] != other.accumulations[dim]:
                    raise ValueError(
                        "Can only merge different accumulations over dim=step"
                    )
                current["accumulations"][dim] = self.accumulations[dim].model_dump(
                    by_alias=True
                )
        return type(self)(**current)
