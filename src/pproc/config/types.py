# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import copy
from typing import Optional, List, Any, Annotated, ClassVar, Iterator
from typing_extensions import Self, Union
from pydantic import model_validator, BaseModel, Field, Tag, Discriminator
import numpy as np
import datetime

from conflator import CLIArg


from pproc.config.base import BaseConfig, Parallelisation
from pproc.config import io
from pproc.config.param import ParamConfig, partial_equality
from pproc.config.utils import _set, _get, update_request, deep_update
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
        quantiles = schema_config.get("quantiles", None)
        return super().from_schema(schema_config, **overrides, quantiles=quantiles)

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
    _merge_exclude = ("name", "inputs", "accumulations")

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
    _merge_exclude = ("total_fields", "parameters")

    def finalise(self):
        # Continue merging until parameters can not be merged anymore
        new_params = self._merge_parameters()
        while new_params != self.parameters:
            self.parameters = new_params
            new_params = self._merge_parameters()
        self.total_fields = 0
        super().finalise()

    def _format_out(self, param: AccumParamConfig, req: dict) -> dict:
        req = req.copy()
        if req["type"] not in ["fcmean", "fcmax", "fcstdev", "fcmin"]:
            return req

        self._append_number(param, req)
        return req

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

    def _merge_total_fields(self, other: Self) -> int:
        # Temporarily set to 1 to avoid validation failure, will be set properly
        # when finalise is called
        return 1


class MonthlyStatsConfig(BaseConfig):
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
    mod: Optional[int] = None
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
                        req["step"] = list(req["step"].values())

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
    clim_total_fields: Annotated[int, Field(validate_default=True)] = 0
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
        use_clim_anomaly = schema_config.pop("use_clim_anomaly", False)
        return super().from_schema(
            schema_config, **overrides, use_clim_anomaly=use_clim_anomaly
        )

    @model_validator(mode="after")
    def validate_totalfields(self) -> Self:
        super().validate_totalfields()
        if self.clim_total_fields == 0:
            self.clim_total_fields = self.compute_totalfields("clim")
        return self

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

    def _merge_clim(self, other: Self) -> None:
        return None


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
    wind: ParamConfig
    cp: ParamConfig
    cdir: ParamConfig
    cape: ParamConfig
    _deps: ClassVar[list[str]] = ["wind", "cp", "cdir", "cape"]
    _merge_exclude = ("accumulations", "wind", "cp", "cdir", "cape")

    @model_validator(mode="before")
    @classmethod
    def validate_deps(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        for param in cls._deps:
            param_config = data[param]
            _set(param_config, "name", param)
        return data

    @property
    def dependencies(self) -> list[ParamConfig]:
        return [getattr(self, dep) for dep in self._deps]

    def in_keys(
        self, inputs: io.InputsCollection, filters: Optional[list[str]] = None
    ) -> Iterator[dict]:
        yield from super().in_keys(inputs, filters)

        for param in self.dependencies:
            yield from param.in_keys(inputs, filters)


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
    min_predictand: float = 0.04

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
        for index, nparam in enumerate(ECPointParamConfig._deps):
            nested_params[nparam] = super()._populate_param(
                config.pop(nparam, {}),
                [inputs_config[index + 1], inputs_config[index + 6]],
                src_name="fc",
                nested=False,
                **overrides.pop(nparam, {}),
            )
        param_config = super()._populate_param(
            config, [inputs_config[0], inputs_config[5]], **overrides
        )
        param_config.update(nested_params)
        return param_config

    def _format_out(self, param: ParamConfig, req) -> dict:
        req = super()._format_out(param, req)
        if req["type"] == "pfc":
            return req

        req.pop("quantile")
        self._append_number(param, req)
        return req
