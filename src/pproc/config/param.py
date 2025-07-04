# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any, Dict, Iterator, Optional
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
from pproc.config.utils import extract_mars, deep_update
from pproc.config.io import Input, InputsCollection
from pproc.config.utils import update_request, expand

logger = logging.getLogger(__name__)


def partial_equality(left: BaseModel, right: BaseModel, exclude: tuple[str]) -> bool:
    left_dict = left.model_dump(by_alias=True, exclude=exclude)
    right_dict = right.model_dump(by_alias=True, exclude=exclude)
    return left_dict == right_dict


class ParamConfig(BaseModel):
    name: str
    inputs: dict = {}
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    accumulations: dict[str, AccumulationConfig] = Field(default_factory=dict)
    dtype_: str = Field(alias="dtype", default="float32")
    metadata: Dict[str, Any] = {}
    total_fields: int = 1
    vod2uv: bool = False
    _merge_exclude: tuple[str] = ("accumulations",)

    @model_validator(mode="after")
    def set_vod2uv(self) -> Self:
        for src_config in self.inputs.values():
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

    def input_list(self, inputs: InputsCollection, name: str, **kwargs) -> list[Input]:
        base_config: Input = getattr(inputs, name)
        config = self.inputs.get(name, {})
        cfg_source = config.get("source", {})
        reqs = update_request(
            copy.deepcopy(base_config.request),
            config.get("request", {}),
            **kwargs,
            **inputs.overrides,
        )

        if self.vod2uv:
            return [
                Input(
                    source={
                        "type": cfg_source.get("type", base_config.type),
                        "path": cfg_source.get("path", base_config.path),
                    },
                    request=reqs,
                )
            ]

        reqs = list(expand(reqs, "param"))
        df = pd.DataFrame(reqs)
        if "param" in df:
            return [
                Input(
                    source={
                        "type": cfg_source.get("type", base_config.type),
                        "path": cfg_source.get("path", base_config.path),
                    },
                    request=[row.dropna().to_dict() for _, row in items.iterrows()],
                )
                for _, items in df.groupby("param")
            ]
        return [
            Input(
                source={
                    "type": cfg_source.get("type", base_config.type),
                    "path": cfg_source.get("path", base_config.path),
                },
                request=reqs,
            )
        ]

    def in_keys(
        self, inputs: InputsCollection, filters: Optional[list[str]] = None
    ) -> Iterator[dict]:
        for input in inputs.names:
            for pinput in self.input_list(inputs, input):
                if filters and pinput.type not in filters:
                    continue

                reqs = (
                    pinput.request
                    if isinstance(pinput.request, list)
                    else [pinput.request]
                )
                for req in reqs:
                    req["source"] = (
                        pinput.path if pinput.path is not None else pinput.type
                    )
                    accum_updates = (
                        getattr(self, input).accumulations
                        if hasattr(self, input)
                        else {}
                    )
                    accumulations = deep_update(
                        self.accumulations.copy(), accum_updates
                    )
                    req.update(
                        {
                            key: accum.unique_coords()
                            for key, accum in accumulations.items()
                        }
                    )
                    yield req

    def out_keys(
        self, inputs: InputsCollection, metadata: Optional[dict] = None
    ) -> Iterator[dict]:
        fc_name = inputs.names[0]
        base_input: Input = getattr(inputs, fc_name)
        param_input = self.inputs.get(fc_name, {})
        reqs = update_request(
            base_input.request,
            param_input.get("request", {}),
            **{
                **inputs.overrides,
                **extract_mars(metadata or {}),
                **extract_mars(self.metadata),
            },
        )
        if isinstance(reqs, dict):
            reqs = [reqs]
        for dim, accum in self.accumulations.items():
            reqs = [{**x, **y} for x, y in itertools.product(reqs, accum.out_mars(dim))]
        yield from reqs

    def _merge_accumulations(self, other: Self) -> dict:
        if self.accumulations.keys() != other.accumulations.keys():
            raise ValueError(
                "Merging of two parameter configs requires them to have the same accumulations dimensions"
            )
        new_accums = {}
        for dim in other.accumulations.keys():
            if dim == "step":
                new_accums[dim] = (
                    self.accumulations[dim]
                    .merge(other.accumulations[dim])
                    .model_dump(by_alias=True)
                )
            else:
                if self.accumulations[dim] != other.accumulations[dim]:
                    raise ValueError(
                        "Can only merge different accumulations over dim=step"
                    )
                new_accums[dim] = self.accumulations[dim].model_dump(by_alias=True)
        return new_accums

    def merge(self, other: Self) -> Self:
        """
        Merge two parameter configurations
        """
        if not partial_equality(self, other, exclude=self._merge_exclude):
            raise ValueError(
                f"Merging of two parameter configs requires them to be the same, except for {self._merge_exclude}"
            )

        merged = self.model_dump(by_alias=True, exclude=self._merge_exclude)
        for attr in self._merge_exclude:
            self_attr = getattr(self, attr)
            if merge_func := getattr(self, f"_merge_{attr}", None):
                merged[attr] = merge_func(other)
            elif isinstance(self_attr, list):
                merged[attr] = self_attr + [
                    x for x in getattr(other, attr) if x not in self_attr
                ]
            elif isinstance(self_attr, ParamConfig):
                merged[attr] = self_attr.merge(getattr(other, attr))
            else:
                raise ValueError(
                    f"No merge protocol defined for {attr} in {type(self)}"
                )
        return type(self)(**merged)
