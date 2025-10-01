# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import copy
import os
from typing import Literal, Optional, List, Any, Annotated, ClassVar, Iterator
from typing_extensions import Self, Union
from pydantic import (
    field_validator,
    model_serializer,
    model_validator,
    BaseModel,
    Field,
    Tag,
    Discriminator,
)
import numpy as np
import datetime

from conflator import CLIArg, ConfigModel
from earthkit.time import DailySequence

from pproc.clustereps.season import MONTH_DAYS, Season
from pproc.config.base import BaseConfig, Parallelisation
from pproc.config import io
from pproc.config.param import ParamConfig, partial_equality
from pproc.config.utils import _set, _get, extract_mars, update_request, deep_update
from pproc.config.preprocessing import Expression
from pproc.common.stepseq import steprange_to_fcmonth
from pproc.extremes.indices import Index, SUPPORTED_INDICES, create_indices


def steprange(steps: list[int] | str) -> str:
    if isinstance(steps, str):
        return steps
    return f"{steps[0]}-{steps[-1]}"


def end_step(step: int | str) -> int:
    return step if isinstance(step, int) else int(step.split("-")[1])


class EnsmsConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.EnsmsOutputModel = io.EnsmsOutputModel()


class QuantilesConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.QuantilesOutputModel = io.QuantilesOutputModel()
    quantiles: int | List[float] = 100
    _total_number: int = 0
    _even_spacing: bool = None

    @property
    def even_spacing(self) -> bool:
        if self._even_spacing is None:
            self._even_spacing = isinstance(self.quantiles, int) or np.all(
                np.diff(self.quantiles) == self.quantiles[1] - self.quantiles[0]
            )
        return self._even_spacing

    @property
    def total_number(self) -> int:
        if self._total_number == 0:
            num_quantiles = (
                self.quantiles
                if isinstance(self.quantiles, int)
                else (len(self.quantiles) - 1)
            )
            self._total_number = num_quantiles if self._even_spacing else 100
        return self._total_number

    def quantile_indices(self, index: int) -> List[int]:
        pert_number = index if self.even_spacing else round(self.quantiles[index] * 100)
        return pert_number, self.total_number

    @classmethod
    def from_schema(cls, schema_config: dict, **overrides) -> Self:
        overrides = overrides.copy()
        if "quantiles" in schema_config:
            overrides.setdefault("quantiles", schema_config.pop("quantiles"))
        return super().from_schema(schema_config, **overrides)

    def _format_out(self, param: ParamConfig, req) -> dict:
        req = super()._format_out(param, req)
        num_quantiles = (
            self.quantiles
            if isinstance(self.quantiles, int)
            else (len(self.quantiles) - 1)
        )
        req["quantile"] = [
            f"{qindices[0]}:{qindices[1]}"
            for qindices in map(self.quantile_indices, range(num_quantiles + 1))
        ]
        return req


class AccumParamConfig(ParamConfig):
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    out_accum_key: str = "perturbationNumber"
    out_accum_values: Optional[list[float]] = None
    _merge_exclude = ("name", "inputs", "accumulations", "total_fields")

    def _merge_inputs(self, other: Self) -> dict:
        if self.inputs == other.inputs:
            return self.inputs
        inputs = copy.deepcopy(self.inputs)
        for key, values in inputs.items():
            requests = values["request"]
            if not isinstance(requests, list):
                requests = [requests]
            other_requests = other.inputs[key]["request"]
            if not isinstance(other_requests, list):
                other_requests = [other_requests]
            inputs[key]["request"] = requests + [
                x for x in other_requests if x not in requests
            ]
        return inputs

    def _merge_name(self, other: Self) -> str:
        return self.name

    def _merge_total_fields(self, other: Self) -> int:
        return 0

    def can_merge(self, other: Self) -> bool:
        if self.accumulations == other.accumulations:
            # Can merge requests of different types e.g. fc and pf if
            # other parts of the source are equal
            compatible_inputs = True
            for src, values in self.inputs.items():
                input = copy.deepcopy(values)
                other_input = copy.deepcopy(other.inputs[src])
                for xinput in [input, other_input]:
                    if isinstance(xinput["request"], dict):
                        xinput["request"] = [xinput["request"]]
                    for req in xinput["request"]:
                        [req.pop(key, None) for key in ["stream", "type", "number"]]
                if input != other_input:
                    compatible_inputs = False
                    break
            if compatible_inputs:
                return True
        return self.inputs == other.inputs


class AccumConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.AccumOutputModel = io.AccumOutputModel()
    parameters: list[AccumParamConfig]
    _merge_exclude = ("parameters",)

    def finalise(self):
        # Continue merging until parameters can not be merged anymore
        new_params = self._merge_parameters()
        while new_params != self.parameters:
            self.parameters = new_params
            new_params = self._merge_parameters()
        super().finalise()

    def _format_out(self, param: AccumParamConfig, req: dict) -> dict:
        req = req.copy()
        if req["type"] not in ["fcmean", "fcmax", "fcstdev", "fcmin"]:
            return req

        self._append_number(param, req)
        return req

    @classmethod
    def _populate_accumulations(cls, inputs: list[dict], base_accum: dict) -> dict:
        accums = super()._populate_accumulations(inputs, base_accum)
        # Allow batching levels
        accums.pop("levelist", None)
        return accums

    def _merge_parameters(self, other: Self = None) -> list[AccumParamConfig]:
        merged_params = [self.parameters[0]]
        other_params = self.parameters[1:]
        if other is not None:
            other_params.extend(other.parameters)
        for in_param in other_params:
            merged = False
            for index, out_param in enumerate(merged_params):
                if out_param.can_merge(in_param):
                    merged_params[index] = out_param.merge(in_param)
                    merged = True
                    break
            if not merged:
                merged_params.append(in_param)
        return merged_params


class MonthlyStatsConfig(AccumConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.MonthlyStatsOutputModel = io.MonthlyStatsOutputModel()
    parameters: list[AccumParamConfig]

    def _format_out(self, param: ParamConfig, req: dict) -> dict:
        req = req.copy()
        step_ranges = req.pop("step")
        date = datetime.datetime.strptime(str(req["date"]), "%Y%m%d")
        fcmonths = [
            steprange_to_fcmonth(date, step_range) for step_range in step_ranges
        ]
        req["fcmonth"] = fcmonths
        return req


class HistParamConfig(ParamConfig):
    bins: List[float]
    mod: Optional[Union[float | int]] = None
    normalise: bool = True
    scale_out: Optional[float] = None

    def _format_out(self, param: ParamConfig, req: dict) -> dict:
        req = super()._format_out(param, req)
        req["quantile"] = [f"{x}:{len(self.bins)}" for x in range(1, len(self.bins))]
        return req


class HistogramConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.HistogramOutputModel = io.HistogramOutputModel()
    parameters: list[HistParamConfig]

    @classmethod
    def from_schema(cls, schema_config: dict, **overrides) -> Self:
        quantiles = schema_config.pop("quantiles", None)
        if not isinstance(quantiles, int):
            quantiles = len(quantiles)
        assert quantiles == len(schema_config["bins"]) - 1
        return super().from_schema(schema_config, **overrides)


class ClimParamConfig(ParamConfig):
    clim: ParamConfig
    _merge_exclude = ("accumulations", "inputs", "clim")

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any) -> Any:
        clim = _get(data, "clim", {})
        if isinstance(clim, dict):
            clim_options = {**data, **clim}
            _set(data, "clim", ParamConfig(**clim_options))
        return data

    def validate_totalfields(self, inputs: io.InputsCollection):
        super().validate_totalfields(inputs)
        if self.clim.total_fields == 0:
            self.clim.total_fields = self.compute_totalfields(inputs, "clim")

    def in_keys(
        self, inputs: io.InputsCollection, filters: Optional[list[str]] = None
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
                    if isinstance(req.get("step", []), dict):
                        req["step"] = list(set(req["step"].values()))
                        req["step"].sort(
                            key=lambda x: x if isinstance(x, int) else x.split("-")[-1]
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
                            if key not in req
                        }
                    )
                    yield req

    def _merge_inputs(self, other: Self) -> dict:
        new_inputs = copy.deepcopy(self.inputs)
        other_inputs = copy.deepcopy(other.inputs)
        if "clim" in new_inputs:
            if "clim" not in other_inputs:
                raise ValueError("Merging of inputs requires same inputs types")
            steps = []
            for input in [new_inputs, other_inputs]:
                clim_request = input["clim"].get("request", {})
                if isinstance(clim_request, list):
                    clim_request = clim_request[0]
                if clim_steps := clim_request.get("step", {}):
                    steps.append(clim_steps)
            if len(steps) > 0:
                if {**steps[0], **steps[1]} != {**steps[1], **steps[0]}:
                    raise ValueError(
                        "Merging of two parameter configs requires clim steps to be compatible"
                    )
                for input in [new_inputs, other_inputs]:
                    updated_request = update_request(
                        input["clim"].get("request", {}),
                        {"step": {**steps[0], **steps[1]}},
                    )
                    input["clim"]["request"] = (
                        updated_request
                        if len(updated_request) > 1
                        else updated_request[0]
                    )

        if new_inputs != other_inputs:
            raise ValueError(
                "Merging of inputs requires equality, except for clim steps"
            )
        return new_inputs


class SigniParamConfig(ClimParamConfig):
    clim_em: ParamConfig
    epsilon: Optional[float] = None
    epsilon_is_abs: bool = True
    _merge_exclude = ("accumulations", "clim", "clim_em")

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any) -> Any:
        clim = _get(data, "clim", {})
        if isinstance(clim, dict):
            clim_options = {**data, **clim}
            _set(data, "clim", ParamConfig(**clim_options))
        clim_em = _get(data, "clim_em", {})
        if isinstance(clim_em, dict):
            if len(clim_em) > 0:
                clim_options = {**data, **clim_em}
            else:
                clim_options = {**data, **clim}
            _set(data, "clim_em", ParamConfig(**clim_options))
        return data


class SigniConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    inputs: io.SignificanceInputModel
    outputs: io.SignificanceOutputModel = io.SignificanceOutputModel()
    parameters: list[SigniParamConfig]
    use_clim_anomaly: Annotated[
        bool,
        CLIArg("--use-clim-anomaly", action="store_true", default=None),
        Field(description="Use anomaly of climatology in significance computation"),
    ] = False

    @classmethod
    def from_schema(cls, schema_config: dict, **overrides) -> Self:
        overrides = overrides.copy()
        if "use_clim_anomaly" in schema_config:
            overrides.setdefault(
                "use_clim_anomaly", schema_config.pop("use_clim_anomaly")
            )
        return super().from_schema(schema_config, **overrides)

    @classmethod
    def sort_inputs(cls, inputs: list[dict]) -> dict:
        sorted_requests = {}
        for inp in inputs:
            is_clim = inp.get("climatology", False)
            if is_clim and inp["type"] == "fcmean":
                src_name = "clim"
            elif is_clim and inp["type"] == "taem":
                src_name = "clim_em"
            else:
                src_name = "fc"
            sorted_requests.setdefault(src_name, []).append(inp)
        return sorted_requests

    @classmethod
    def _populate_param(
        cls,
        config: dict,
        inputs_config,
        src_name: Optional[str] = None,
        nested: bool = False,
        **overrides,
    ) -> dict:
        nested_params = {}
        for nparam in ["clim", "clim_em"]:
            nested_params[nparam] = super()._populate_param(
                config.pop(nparam, {}),
                inputs_config,
                src_name=nparam,
                nested=True,
                **overrides.pop(nparam, {}),
            )
        param_config = super()._populate_param(config, inputs_config, **overrides)
        param_config.update(nested_params)
        return param_config


class AnomalyConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    inputs: io.ClimInputModel
    outputs: io.AnomalyOutputModel = io.AnomalyOutputModel()
    parameters: list[ClimParamConfig]

    @classmethod
    def _populate_param(
        cls,
        config: dict,
        inputs_config,
        src_name: Optional[str] = None,
        nested: bool = False,
        **overrides,
    ) -> dict:
        nested_params = {}
        for nparam in ["clim"]:
            nested_params[nparam] = super()._populate_param(
                config.pop(nparam, {}),
                inputs_config,
                src_name=nparam,
                nested=True,
                **overrides.pop(nparam, {}),
            )
        param_config = super()._populate_param(config, inputs_config, **overrides)
        param_config.update(nested_params)
        return param_config

    @classmethod
    def sort_inputs(cls, inputs: list[dict]) -> dict:
        sorted_requests = {}
        for inp in inputs:
            is_clim = inp.get("climatology", False)
            if is_clim:
                src_name = "clim"
            else:
                src_name = "fc"
            sorted_requests.setdefault(src_name, []).append(inp)
        return sorted_requests

    def _format_out(self, param: ClimParamConfig, req: dict) -> dict:
        req = req.copy()
        if req["type"] != "fcmean":
            req.pop("number", None)
            return req

        self._append_number(param, req)
        return req


def anom_discriminator(config: Any) -> str:
    clim = _get(config, "clim", None)
    return "clim" if clim else "base"


class ProbParamConfig(ClimParamConfig):
    clim: Optional[ParamConfig] = None
    _merge_exclude = ("name", "accumulations", "inputs", "clim")

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any) -> Any:
        clim = _get(data, "clim", None)
        if isinstance(clim, dict):
            clim_options = {
                **data,
                "preprocessing": [],
                "accumulations": {},
                "metadata": {},
                **clim,
            }
            _set(data, "clim", ParamConfig(**clim_options))
        return data

    def validate_totalfields(self, inputs: io.InputsCollection):
        if self.total_fields == 0:
            self.total_fields = self.compute_totalfields(inputs)
        if self.clim is not None and self.clim.total_fields == 0:
            self.clim.total_fields = self.compute_totalfields(inputs, "clim")

    def _merge_clim(self, other: Self) -> None:
        return None

    def _merge_name(self, other: Self) -> str:
        return self.name

    def can_merge(self, other: Self) -> bool:
        return self.name == other.name or self.inputs == other.inputs


class ProbConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    inputs: Annotated[
        Union[
            Annotated[io.BaseInputModel, Tag("base")],
            Annotated[io.ClimInputModel, Tag("clim")],
        ],
        Discriminator(anom_discriminator),
    ]
    outputs: io.ProbOutputModel = io.ProbOutputModel()
    parameters: list[ProbParamConfig]

    @model_validator(mode="after")
    def validate_param(self) -> Self:
        if isinstance(self.inputs, io.ClimInputModel):
            for param in self.parameters:
                if not param.clim:
                    param.clim = ParamConfig(
                        **param.model_dump(
                            exclude=("preprocessing", "accumulations", "metadata"),
                            by_alias=True,
                        ),
                        accumulations={},
                    )
        return self

    @classmethod
    def from_schema(cls, schema_config: dict, **overrides) -> Self:
        schema_config = copy.deepcopy(schema_config)
        threshold = schema_config.pop("threshold")
        threshold["out_paramid"] = schema_config["metadata"].pop("paramId")
        schema_config["accumulations"]["step"]["thresholds"] = [threshold]
        return super().from_schema(schema_config, **overrides)

    @classmethod
    def _input_request(
        cls, src_name: str, requests: list[dict], accum_dims: list[str], **overrides
    ) -> dict | list[dict]:
        if src_name == "clim":
            accum_dims = accum_dims.copy()
            accum_dims.remove("step")
        return super()._input_request(src_name, requests, accum_dims, **overrides)

    @classmethod
    def sort_inputs(cls, inputs: list[dict]) -> dict:
        sorted_requests = {}
        fc_step: list[int]
        clim_step: Optional[list[int]] = None
        for inp in inputs:
            steps = inp["step"] if isinstance(inp["step"], list) else [inp["step"]]
            is_clim = inp.get("climatology", False)
            if is_clim:
                src_name = "clim"
                clim_step = steps
            else:
                src_name = "fc"
                fc_step = steps
            sorted_requests.setdefault(src_name, []).append(inp.copy())

        if clim_step is not None:
            assert len(fc_step) == len(
                clim_step
            ), f"Forecast and clim steps must be of the same length"
            for clim_inp in sorted_requests.get("clim", []):
                clim_inp["step"] = {
                    fc_step[x]: clim_step[x] for x in range(len(fc_step))
                }
        return sorted_requests

    def _merge_parameters(self, other: Self = None) -> list[ProbParamConfig]:
        merged_params = [self.parameters[0]]
        other_params = self.parameters[1:]
        if other is not None:
            other_params.extend(other.parameters)
        for in_param in other_params:
            merged = False
            for index, out_param in enumerate(merged_params):
                if out_param.can_merge(in_param):
                    merged_params[index] = out_param.merge(in_param)
                    merged = True
                    break
            if not merged:
                merged_params.append(in_param)
        return merged_params


class ExtremeParamConfig(ClimParamConfig):
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    eps: float = -1.0
    sot: list[int] = []
    cpf_eps: Optional[float] = None
    cpf_symmetric: bool = False
    compute_indices: list[str] = ["efi", "sot"]
    allow_grib1_to_grib2: bool = False
    _merge_exclude: tuple[str] = (
        "accumulations",
        "inputs",
        "clim",
        "sot",
        "cpf_eps",
        "compute_indices",
    )

    @model_validator(mode="after")
    def validate_indices(self) -> Self:
        for index in self.compute_indices:
            if index not in SUPPORTED_INDICES:
                raise ValueError(
                    f"Unsupported index {index}. Supported indices are {SUPPORTED_INDICES}"
                )
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any) -> Any:
        clim = _get(data, "clim", {})
        if isinstance(clim, dict):
            clim_options = {
                **data,
                "preprocessing": [],
                "accumulations": {},
                "metadata": {},
                **clim,
            }
            _set(data, "clim", ParamConfig(**clim_options))
        return data

    @property
    def indices(self) -> dict[str, Index]:
        return create_indices(self.compute_indices, self.model_dump())

    def out_keys(
        self, inputs: io.InputsCollection, metadata: Optional[dict] = None
    ) -> Iterator:
        base_outs = [req for req in super().out_keys(inputs, metadata)]
        indices = self.compute_indices.copy()
        if np.any([x["type"] in ["cf", "fc"] for x in base_outs]):
            indices.append("efic")
        req = base_outs[0].copy()
        for index in indices:
            if index == "sot" and len(self.sot) == 0:
                continue
            req["type"] = index
            yield req

    def _merge_clim(self, other: Self) -> dict:
        return {}

    def _merge_cpf_eps(self, other: Self) -> Optional[float]:
        if self.cpf_eps is None:
            return other.cpf_eps

        if other.cpf_eps is not None and other.cpf_eps != self.cpf_eps:
            raise ValueError(
                "Merging of parameter configs requires cpf_eps to be equal"
            )

        return self.cpf_eps


class ExtremeConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    inputs: io.ClimInputModel
    outputs: io.ExtremeOutputModel = io.ExtremeOutputModel()
    parameters: list[ExtremeParamConfig]

    def _format_out(self, param: ParamConfig, req: dict) -> dict:
        req = super()._format_out(param, req)
        if req["type"] == "sot":
            req["number"] = param.sot
        return req

    @classmethod
    def _input_request(
        cls, src_name: str, requests: list[dict], accum_dims: list[str], **overrides
    ) -> dict | list[dict]:
        if src_name == "clim":
            accum_dims = accum_dims.copy()
            accum_dims.remove("step")
        return super()._input_request(src_name, requests, accum_dims, **overrides)

    @classmethod
    def sort_inputs(cls, inputs: list[dict]) -> dict:
        sorted_requests = {}
        fc_step = 0
        clim_step = 0
        for inp in inputs:
            is_clim = inp.get("climatology", False)
            if is_clim:
                src_name = "clim"
                clim_step = inp["step"]
            else:
                src_name = "fc"
                fc_step = steprange(inp["step"])
            sorted_requests.setdefault(src_name, []).append(inp.copy())

        for clim_inp in sorted_requests.get("clim", []):
            clim_inp["step"] = {fc_step: clim_step}
        return sorted_requests


class WindConfig(BaseConfig):
    parallelisation: int = 1
    outputs: io.WindOutputModel = io.WindOutputModel()
    parameters: list[ParamConfig]

    def _format_out(self, param: ParamConfig, req: dict) -> dict:
        req = req.copy()
        if req["type"] in ["em", "es"]:
            req.pop("number", None)
        return req


class ThermoParamConfig(ParamConfig):
    out_params: list[str | int]

    def in_keys(
        self, inputs: io.InputsCollection, filters: Optional[list[str]] = None
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
                    req.update(
                        {
                            key: accum.unique_coords()
                            for key, accum in self.accumulations.items()
                        }
                    )
                    # Override step for instantaneous params, which is equal to output steps
                    if input == "inst":
                        req["step"] = list(self.out_keys(inputs))[0]["step"]
                    yield req

    def out_keys(
        self, inputs: io.InputsCollection, metadata: Optional[dict] = None
    ) -> Iterator:
        for req in super().out_keys(inputs, metadata):
            req["param"] = self.out_params
            req["step"] = [end_step(x) for x in req["step"]]
            yield req

    def _merge_inputs(self, other: Self) -> dict:
        # inst + inst -> merge accums and input params
        # accum + accum -> merge accums and input params
        # inst + (inst, accum) -> only merge in inst steps are encompassed in accum step ranges, becomes accum
        new_inputs = copy.deepcopy(self.inputs)
        other_inputs = copy.deepcopy(other.inputs)
        for key in new_inputs:
            if key in other_inputs:
                current_params = new_inputs[key]["request"].pop("param")
                other_params = other_inputs[key]["request"].pop("param")
                if not isinstance(current_params, list):
                    current_params = [current_params]
                if not isinstance(other_params, list):
                    other_params = [other_params]
                if new_inputs[key] != other_inputs[key]:
                    raise ValueError(
                        "Only inputs equal up to request param can be merged"
                    )
                new_inputs[key]["request"]["param"] = current_params + [
                    x for x in other_params if x not in current_params
                ]
        for key in other_inputs:
            if key not in new_inputs:
                new_inputs[key] = other_inputs[key]
        return new_inputs

    def can_merge(self, other: Self) -> bool:
        if self.out_params == other.out_params:
            return True
        if self.accumulations == other.accumulations:
            return True
        out_steps = sum(
            [
                [end_step(x) for x in steps["step"]]
                for steps in self.accumulations["step"].out_mars("step")
            ],
            [],
        )
        other_steps = sum(
            [
                [end_step(x) for x in steps["step"]]
                for steps in other.accumulations["step"].out_mars("step")
            ],
            [],
        )
        return out_steps == other_steps

    def merge(self, other: Self) -> Self:
        if self.out_params == other.out_params:
            return super().merge(other)
        exclude = ("name", "accumulations", "out_params", "inputs")
        if not partial_equality(self, other, exclude=exclude):
            raise ValueError(
                f"Merging of two parameter configs requires equality, except for {exclude}"
            )

        merged = self.model_dump(by_alias=True, exclude=exclude)
        if (
            self.accumulations != other.accumulations
            and self.accumulations["step"].operation != "difference"
        ):
            merged["accumulations"] = other.accumulations
        else:
            merged["accumulations"] = self.accumulations
        merged["out_params"] = self.out_params + [
            x for x in other.out_params if x not in self.out_params
        ]
        merged["inputs"] = self._merge_inputs(other)
        merged["name"] = self.name
        return type(self)(**merged)


class ThermoConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    inputs: io.ThermoInputModel
    outputs: io.ThermoOutputModel = io.ThermoOutputModel()
    parameters: list[ThermoParamConfig]
    validateutci: bool = False
    utci_misses: bool = False
    _merge_exclude: tuple[str] = ("parameters", "inputs")

    @model_validator(mode="after")
    def check_params(self) -> Self:
        # Output of config generation can have additional
        # parameters, which can be merged. This ensures they are merged
        # as soon as possible
        new_params = self._merge_parameters()
        if new_params != self.parameters:
            self.parameters = new_params
        return self

    @classmethod
    def from_schema(cls, schema_config: dict, **overrides) -> Self:
        paramId = schema_config["metadata"].pop("paramId")
        schema_config["out_params"] = [paramId]

        outputs = overrides.setdefault("outputs", {})
        for out_name in io.ThermoOutputModel.names:
            if out_name != "indices" and out_name not in outputs:
                outputs["out_name"] = {"target": {"type": "null"}}
        return super().from_schema(schema_config, **overrides)

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        for param in self.parameters:
            inputs = param.input_list(self.inputs, "accum")
            if any([src.type == "null" for src in inputs]):
                for out_req in param.accumulations["step"].out_mars("step"):
                    steps = out_req["step"]
                    if isinstance(steps, (str, int)):
                        steps = [steps]
                    nsteps = list(map(lambda x: len(str(x).split("-")), steps))
                    assert np.all(
                        np.asarray(nsteps) == 1
                    ), f"Accumulation inputs required for step ranges."
        return self

    @classmethod
    def sort_inputs(cls, inputs: list[dict]) -> dict:
        sorted_requests = {}
        for inp in inputs:
            if isinstance(inp["step"], list) and len(inp["step"]) > 1:
                src_name = "accum"
            else:
                src_name = "inst"
            sorted_requests.setdefault(src_name, []).append(inp)
        return sorted_requests

    def _merge_parameters(self, other: Self = None) -> list[ThermoParamConfig]:
        merged_params = [self.parameters[0]]
        other_params = self.parameters[1:]
        if other is not None:
            other_params.extend(other.parameters)
        for in_param in other_params:
            merged = False
            for index, out_param in enumerate(merged_params):
                if out_param.can_merge(in_param):
                    merged_params[index] = out_param.merge(in_param)
                    merged = True
                    break
            if not merged:
                merged_params.append(in_param)
        return merged_params

    def _merge_inputs(self, other: Self) -> io.ThermoInputModel:
        new_inputs = self.inputs.model_copy()
        other_inputs = other.inputs.model_copy()
        if new_inputs.accum.type == "null":
            new_inputs.accum.type = other_inputs.accum.type
            new_inputs.accum.path = other_inputs.accum.path
        if other_inputs.accum.type == "null" and new_inputs.accum.type != "null":
            other_inputs.accum.type = new_inputs.accum.type
            other_inputs.accum.path = new_inputs.accum.path
        if new_inputs != other_inputs:
            raise ValueError(
                "Can only merge configs with inputs differing by accum input type"
            )
        return new_inputs

    def _format_out(self, param: ParamConfig, req) -> dict:
        req = super()._format_out(param, req)
        if req["type"] in ["cf", "fc"]:
            return req
        self._append_number(param, req)
        return req


class ECPointParamConfig(ParamConfig):
    dependencies: dict[str, ParamConfig]
    num_inputs: int
    _merge_exclude = ("accumulations", "dependencies")

    @model_validator(mode="before")
    @classmethod
    def validate_deps(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        for name, param in data.get("dependencies", {}).items():
            _set(param, "name", name)
        return data

    def in_keys(
        self, inputs: io.InputsCollection, filters: Optional[list[str]] = None
    ) -> Iterator[dict]:
        yield from super().in_keys(inputs, filters)

        for param in self.dependencies.values():
            yield from param.in_keys(inputs, filters)

    def _merge_dependencies(self, other: Self) -> dict[str, ParamConfig]:
        new_deps = {}
        for name, config in self.dependencies.items():
            new_deps[name] = config.merge(other.dependencies[name])
        return new_deps

    def validate_totalfields(self, inputs: io.InputsCollection):
        for input_config in [self] + list(self.dependencies.keys()):
            if isinstance(input_config, str):
                input_config = param.dependencies[input_config]
            input_config.validate_totalfields(inputs)


class ECPointParallelisation(BaseModel):
    n_par_read: int = 1
    wt_batch_size: int = 1
    ens_batch_size: int = 1


class ECPointConfig(QuantilesConfig):
    parallelisation: ECPointParallelisation = ECPointParallelisation()
    outputs: io.ECPointOutputModel = io.ECPointOutputModel()
    parameters: list[ECPointParamConfig]
    bp_location: Annotated[
        str, CLIArg("--bp-loc"), Field(description="Location of BP CSV file")
    ]
    fer_location: Annotated[
        str, CLIArg("--fer-loc"), Field(description="Location of FER CSV file")
    ]
    predictors: list[Union[str, Expression]]
    predictant: str
    min_predictant: Optional[float] = None
    scale_outputs: Optional[float] = None

    @classmethod
    def from_schema(cls, schema_config: dict, **overrides) -> Self:
        overrides = overrides.copy()
        for var in [
            "bp_location",
            "fer_location",
            "predictors",
            "predictant",
            "min_predictant",
            "scale_outputs",
        ]:
            if var in schema_config:
                overrides.setdefault(var, schema_config.pop(var))
        return super().from_schema(schema_config, **overrides)

    @classmethod
    def _populate_param(
        cls,
        config: dict,
        inputs_config,
        src_name: Optional[str] = None,
        nested: bool = False,
        **overrides,
    ) -> dict:
        paired_requests = {}
        for inp in inputs_config:
            param = inp["param"] if isinstance(inp["param"], str) else inp["param"][0]
            paired_requests.setdefault(param, []).append(inp)
        paired_requests = list(paired_requests.values())

        dependencies = {}
        config_dep = config.pop("dependencies")
        for index, (name, param_config) in enumerate(config_dep.items()):
            if "dtype" in config:
                param_config.setdefault("dtype", config["dtype"])
            dependencies[name] = super()._populate_param(
                param_config,
                paired_requests[index + 1],
                src_name="fc",
                nested=False,
                **overrides,
            )
        param_config = super()._populate_param(config, paired_requests[0], **overrides)
        param_config["dependencies"] = dependencies
        return param_config

    @classmethod
    def _populate_accumulations(cls, inputs: list[dict], base_accum: dict) -> dict:
        if base_accum is None:
            return {}
        return super()._populate_accumulations(inputs, base_accum)

    def _format_out(self, param: ParamConfig, req) -> dict:
        req = super()._format_out(param, req)
        req["model"] = "ecPoint"
        if req["type"] == "pfc":
            return req

        req.pop("quantile")
        self._append_number(param, req)
        return req


class BoundingBox(ConfigModel):
    lat_n: float
    lat_s: float
    lon_w: float
    lon_e: float

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.lat_n, self.lat_s, self.lon_w, self.lon_e)


class ClusterBaseConfig(BaseConfig):
    # TODO: replace with window {
    step_start: int
    step_end: int
    step_del: int
    # }
    bbox: BoundingBox
    # TODO: wrap in param config {
    num_members: int
    max_anom: float = 10000.0
    metadata: dict[str, Any] = {}
    # }
    parameters: list = Field(default_factory=list)

    @property
    def steps(self) -> list[int]:
        return list(range(self.step_start, self.step_end + 1, self.step_del))

    @property
    def clip(self) -> float:  # FIXME: remove eventually
        return self.max_anom


class ClusterPCAConfig(ClusterBaseConfig):
    num_components: int
    pca_factor: float | None = None


class ClusterPCAStandaloneConfig(ClusterPCAConfig):
    inputs: io.ClusterInputModel
    output: Annotated[
        str,
        CLIArg("-o", "--output", required=True),
        Field(description="Ouptut file (NPZ)"),
    ]


class ClusterClusterConfig(ClusterBaseConfig):
    # Variance threshold
    var_th: float
    # Number of PCs to use, optional
    npc: int = -1
    # Normalisation factor (2/5)
    cluster_factor: float = 0.4
    # Max number of clusters
    ncl_max: int
    # Number of clustering passes
    npass: int
    # Number of red-noise samples for significance computation
    nrsamples: int
    # Maximum significance threshold
    max_sig: float
    # Medium significance threshold
    med_sig: float
    # Minimum significance threshold
    min_sig: float
    # Significance tolerance
    sig_tol: float
    # Parallel red-noise sampling
    n_par: int = 1
    # Initialisation method (k-means++ or sector)
    init: Literal["k-means++", "sector"]

    indexes: Annotated[
        Optional[str],
        CLIArg("-I", "--indexes"),
        Field(description="Cluster indexes output (NPZ)"),
    ] = None
    ncomp_file: Annotated[
        Optional[str],
        CLIArg("-N", "--ncomp-file"),
        Field(description="Number of components output (text)"),
    ] = None
    deterministic_is_control: Annotated[
        bool,
        CLIArg("--deterministic-is-control", action="store_true", default=None),
    ] = False

    @field_validator("npc", mode="before")
    @classmethod
    def npc_from_file(cls, value: Any):
        if isinstance(value, str) and os.path.isfile(value):
            with open(value, "r") as f:
                value = int(f.read().strip())
        return value


class ClusterClusterStandaloneConfig(ClusterClusterConfig):
    pca: Annotated[
        str,
        CLIArg("-p", "--pca", required=True),
        Field(description="PCA data (NPZ)"),
    ]
    template: Annotated[
        str,
        CLIArg("-t", "--template", required=True),
        Field(description="Field to extract keys from (GRIB)"),
    ]

    outputs: io.ClusterClusterOutputModel = io.ClusterClusterOutputModel()


class SeasonConfig(ConfigModel):
    months: list[tuple[int, int]]

    @model_validator(mode="before")
    @classmethod
    def from_months(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"months": data}
        return data

    @model_serializer
    def serialise_model(self):
        return [[start, end] for start, end in self.months]

    def get_season(self, date: datetime.datetime) -> Season:
        month = date.month
        for start, end in self.months:
            if start <= month <= end:
                return Season(start, end, date.year)
            if end < start:
                if month <= end:
                    return Season(start, end, date.year)
                if start <= month:
                    return Season(start, end, date.year + 1)
        raise ValueError(f"No season containing month {month}")


class ClusterAttributionConfig(ClusterBaseConfig):
    ncl_clim: int = 6
    clim_means_: str = Field(alias="clim_means")
    clim_pcs_: str = Field(alias="clim_pcs")
    clim_sdv_: str = Field(alias="clim_sdv")
    clim_eof_: str = Field(alias="clim_eof")
    clim_cluster_index_: str = Field(alias="clim_cluster_index")
    clim_cluster_centroids_eof_: str = Field(alias="clim_cluster_centroids_eof")
    seasons: SeasonConfig = Field(
        default_factory=(lambda: SeasonConfig(months=[(1, 12)]))
    )

    date: Annotated[
        datetime.datetime,
        CLIArg("--date", metavar="YYYYMMDD"),
        Field(description="Forecast date (YYYYMMDD)"),
    ]
    clim_dir: Annotated[
        str,
        CLIArg("--clim-dir", metavar="DIR"),
        Field(description="Climatological data root directory"),
    ]
    output_root: Annotated[
        str,
        CLIArg("-o", "--output-root", metavar="DIR"),
        Field(description="Output directory for reports", default_factory=os.getcwd),
    ]

    @field_validator("date", mode="before")
    @classmethod
    def validate_date(cls, value: Any) -> datetime.datetime:
        if isinstance(value, datetime.datetime):
            return value
        elif isinstance(value, datetime.date):
            return datetime.datetime.combine(value, datetime.time())
        elif isinstance(value, str):
            return datetime.datetime.strptime(value, "%Y%m%d")
        else:
            raise ValueError(f"Cannot parse date from {value!r}")

    @property
    def step_date(self) -> datetime.datetime:  # FIXME: remove eventually?
        return self.date + datetime.timedelta(hours=self.step_start)

    @property
    def this_season(self) -> Season:  # FIXME: remove eventually?
        return self.seasons.get_season(self.date)

    @property
    def month_start_dos(self) -> int:  # FIXME: remove eventually?
        return self.this_season.dos(self.date.replace(day=1))

    @property
    def month_end_dos(self) -> int:  # FIXME: remove eventually?
        return self.this_season.dos(
            self.date.replace(day=MONTH_DAYS[self.date.month] - 1)
        )

    @property
    def clim_means(self) -> str:
        return os.path.join(self.clim_dir, self.clim_means_)

    @property
    def clim_pcs(self) -> str:
        val = os.path.join(self.clim_dir, self.clim_pcs_)
        return val.format(season=self.this_season.name)

    @property
    def clim_sdv(self) -> str:
        val = os.path.join(self.clim_dir, self.clim_sdv_)
        return val.format(season=self.this_season.name)

    @property
    def clim_eof(self) -> str:
        val = os.path.join(self.clim_dir, self.clim_eof_)
        return val.format(season=self.this_season.name)

    @property
    def clim_cluster_index(self) -> str:
        val = os.path.join(self.clim_dir, self.clim_cluster_index_)
        return val.format(season=self.this_season.name)

    @property
    def clim_cluster_centroids_eof(self) -> str:
        val = os.path.join(self.clim_dir, self.clim_cluster_centroids_eof_)
        return val.format(season=self.this_season.name)


class ClusterAttributionStandaloneConfig(ClusterAttributionConfig):
    inputs: io.ClusterAttributionInputModel
    outputs: io.ClusterOutputModel = io.ClusterOutputModel()


class ClusterFullConfig(
    ClusterAttributionConfig, ClusterClusterConfig, ClusterPCAConfig
):
    generate_dummy: bool = False
    inputs: io.ClusterInputModel
    outputs: io.ClusterOutputModel = io.ClusterOutputModel()

    compute_spread: Annotated[
        bool,
        CLIArg("--compute-spread", action="store_true", default=None),
        Field(description="Compute spread from the given source"),
    ] = False

    pca_output: Annotated[
        Optional[str],
        CLIArg("-P", "--pca"),
        Field(description="PCA outputs (NPZ)"),
    ] = None

    _merge_exclude = ("inputs",)

    @property
    def ncl_dummy(self) -> int | None:
        if self.generate_dummy:
            return self.ncl_max
        return None

    @staticmethod
    def _derive_num_members(inputs: list[dict[str, object]]) -> int:
        num = 0
        for input in inputs:
            numbers = input.get("number", 0)
            num_loc = 0
            if isinstance(numbers, (list, range)):
                num_loc = len(numbers)
            elif isinstance(numbers, str):
                num_loc = numbers.count("/") + 1
            else:
                num_loc = 1
            num += num_loc
        return num

    @staticmethod
    def _derive_steps(inputs: list[dict[str, object]]) -> tuple[int, int]:
        starts = set()
        ends = set()
        for input in inputs:
            if "step" not in input:
                continue
            steps = input["step"]
            if isinstance(steps, str):
                steps = [int(s) for s in steps.split("/")]
            starts.add(min(steps))
            ends.add(max(steps))
        assert len(starts) == 1, "Inconsistent start steps"
        assert len(ends) == 1, "Inconsistent end steps"
        return starts.pop(), ends.pop()

    @staticmethod
    def _derive_date(inputs: list[dict[str, object]]) -> str:
        dates = set(input["date"] for input in inputs if "date" in input)
        assert len(dates) == 1, "Inconsistent input dates"
        date = dates.pop()
        return date

    @staticmethod
    def _derive_spread_req(
        inputs: list[dict[str, object]], ref_type: str = "pf", spread_type: str = "es"
    ) -> list[dict[str, object]]:
        reqs = []
        for req in inputs:
            if req.get("type", "") != ref_type:
                continue
            new_req = {**req, "type": spread_type}
            new_req.pop("number", None)
            reqs.append(new_req)
        return reqs

    @classmethod
    def from_schema(cls, schema_config: dict, **overrides) -> Self:
        import yaml

        with open("_schema_config.yaml", "w") as f:
            yaml.safe_dump(schema_config, f)
        schema_config = copy.deepcopy(schema_config)
        outputs = schema_config.pop("outputs", {})
        overrides = copy.deepcopy(overrides)
        overrides.pop("parameters")

        # Construct parameter config
        schema_inputs = copy.deepcopy(schema_config.pop("inputs"))
        interp_keys = schema_config.pop("interp_keys", {})
        for req in schema_inputs:
            if grid := req.pop("target_grid", None):
                req["interpolate"] = {
                    "grid": grid,
                    **interp_keys,
                }

        num_members = cls._derive_num_members(schema_inputs)
        step_start, step_end = cls._derive_steps(schema_inputs)
        date = cls._derive_date(schema_inputs)

        inputs = {
            "fc": {"source": {"type": "fdb"}, "request": schema_inputs},
            "spread": {
                "source": {"type": "fdb"},
                "request": cls._derive_spread_req(schema_inputs),
            },
        }

        config = {
            **schema_config,
            "num_members": num_members,
            "step_start": step_start,
            "step_end": step_end,
            "date": date,
            "inputs": inputs,
            "outputs": deep_update({"default": {"target": {"type": "fdb"}}}, outputs),
            "parameters": {},
        }
        deep_update(config, overrides)
        with open("_gen_config.yaml", "w") as f:
            yaml.safe_dump(config, f)
        return cls(**config)

    def _merge_inputs(self, other: Self) -> io.ClusterInputModel:
        if self.inputs != other.inputs:
            raise ValueError("Cannot merge different input configurations")
        return self.inputs

    def in_mars(self, sources: Optional[list[str]] = None) -> Iterator:
        steps = list(range(self.step_start, self.step_end + 1, self.step_del))
        spread_dates = [
            date.strftime("%Y%m%d")
            for date in DailySequence().range(
                self.date - datetime.timedelta(days=31), self.date, include_end=False
            )
        ]
        acc_keys = {
            "fc": {"step": steps},
            "spread": {"date": spread_dates, "step": steps},
            "deterministic": {"step": steps},
        }
        seen = set()
        for input in self.inputs.names:
            pinput = getattr(self.inputs, input)

            if sources and pinput.type not in sources:
                continue
            if pinput.type == "null":
                continue

            reqs = copy.deepcopy(
                pinput.request if isinstance(pinput.request, list) else [pinput.request]
            )
            for req in reqs:
                req["source"] = pinput.path if pinput.path is not None else pinput.type
                req.update(acc_keys[input])
                req.pop("interpolate", None)
                if str(req) not in seen:
                    seen.add(str(req))
                    yield req

    def out_mars(self, targets: Optional[list[str]] = None) -> Iterator:
        outputs = []
        for name in self.outputs.names:
            if name == "default":
                continue
            output = getattr(self.outputs, name)
            out_type = output.target.type_
            if out_type == "null" or (targets and out_type not in targets):
                continue
            outputs.append(output)

        seen = []
        for output in outputs:
            fc_reqs = self.inputs.fc.request
            reqs = copy.deepcopy(fc_reqs if isinstance(fc_reqs, list) else [fc_reqs])
            for req in reqs:
                req.update(output.metadata)
                req["target"] = (
                    output.target.path
                    if hasattr(output.target, "path")
                    else output.target.type_
                )
                req.update(extract_mars(self.outputs.overrides))
                req["step"] = range(self.step_start, self.step_end + 1, self.step_del)
                req["number"] = range(1, self.ncl_max + 1)
                req["domain"] = "h"
                req.pop("interpolate", None)
                if req not in seen:
                    seen.append(req)
                    yield req
