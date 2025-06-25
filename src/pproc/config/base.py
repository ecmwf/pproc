# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any, Iterator, Optional
import yaml
import itertools
import numpy as np
import copy

from annotated_types import Annotated
from conflator import CLIArg, ConfigModel
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from pproc.config import io
from pproc.config.log import LoggingConfig
from pproc.config.param import ParamConfig, partial_equality
from pproc.config.utils import deep_update, extract_mars, update_request, _get, _set


class Parallelisation(ConfigModel):
    n_par_read: int = 1
    n_par_compute: int = 1
    queue_size: int

    @model_validator(mode="before")
    @classmethod
    def validate_queue(cls, data: Any) -> Any:
        if not _get(data, "queue_size", None):
            _set(data, "queue_size", _get(data, "n_par_compute", 1))
        return data


class Recovery(ConfigModel):
    enable_checkpointing: bool = True
    from_checkpoint: Annotated[
        bool,
        CLIArg("--recover", action="store_true", default=None),
        Field(description="Recover from checkpoint"),
    ] = False
    root_dir: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: Any) -> Any:
        if isinstance(data, bool) and data:
            return {"enable_checkpointing": True, "from_checkpoint": {True}}
        return data

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.from_checkpoint:
            if not self.enable_checkpointing:
                raise ValueError(
                    "Cannot recover from checkpoint without enabling checkpointing"
                )
        return self


class BaseConfig(ConfigModel):
    log: LoggingConfig = LoggingConfig()
    total_fields: Annotated[int, Field(validate_default=True)] = 0
    parallelisation: int | Parallelisation = 1
    recovery: Recovery = Recovery()
    inputs: io.BaseInputModel
    outputs: io.BaseOutputModel = io.BaseOutputModel()
    parameters: list[ParamConfig]
    _init: bool = False
    _merge_exclude: tuple[str] = ("parameters",)

    def print(self):
        print(yaml.dump(self.model_dump(by_alias=True), sort_keys=False))

    @model_validator(mode="after")
    def _init_targets(self) -> Self:
        if self._init:
            return self

        for name in self.outputs.names:
            target = getattr(self.outputs, name).target
            if (isinstance(self.parallelisation, int) and self.parallelisation > 1) or (
                isinstance(self.parallelisation, Parallelisation)
                and self.parallelisation.n_par_compute > 1
            ):
                target.enable_parallel()
            if self.recovery.from_checkpoint:
                target.enable_recovery()
        self._init = True
        return self

    def compute_totalfields(self, src_name: str) -> int:
        out = 0
        for param in self.parameters:
            total_fields = 0
            inputs = param.input_list(self.inputs, src_name)
            reqs = inputs[0].request
            if isinstance(reqs, dict):
                reqs = [reqs]
            for req in reqs:
                if len(req) == 0:
                    continue
                if isinstance(req.get("number", None), list):
                    total_fields += len(req["number"])
                else:
                    total_fields += 1
            if out == 0:
                out = total_fields
            elif out != total_fields:
                raise ValueError(
                    f"All parameters must request the same number of total fields. Expected {out}, got {total_fields}."
                )
        assert out != 0, ValueError("Could not derived total_fields from requests.")
        return out

    @model_validator(mode="after")
    def validate_totalfields(self) -> Self:
        if self.total_fields == 0 and len(self.parameters) > 0:
            total_fields = self.compute_totalfields(self.inputs.names[0])
            self.total_fields = total_fields
        return self

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_params(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = [{"name": name, **param} for name, param in data.items()]
        return data

    def param(self, name: str) -> ParamConfig | None:
        for param in self.parameters:
            if param.name == name:
                return param
        return None

    @classmethod
    def from_schema(cls, schema_config: dict, **overrides) -> Self:
        schema_config = copy.deepcopy(schema_config)
        overrides = copy.deepcopy(overrides)

        # Construct parameter config
        inputs = copy.deepcopy(schema_config.pop("inputs"))
        interp_keys = schema_config.pop("interp_keys", {})
        for req in inputs:
            if grid := req.pop("interp_grid", None):
                req["interpolate"] = {
                    "grid": grid,
                    **interp_keys,
                }

        param_name = schema_config.pop("name", str(inputs[0]["param"]))
        all_param_overrides = overrides.pop("parameters", {})
        param_overrides = all_param_overrides.get(
            param_name, all_param_overrides.get("default", {})
        )
        param_config = cls._populate_param(schema_config, inputs, **param_overrides)

        config = {
            "inputs": {
                src: {"source": {"type": "fdb"}}
                for src in param_config["inputs"].keys()
            },
            "outputs": {"default": {"target": {"type": "fdb"}}},
            "parameters": {param_name: param_config},
        }
        deep_update(config, overrides)
        return cls(**config)

    def merge(self, other: Self, finalise: bool = True) -> Self:
        """
        Merge two configs, where all elements except for parameters must be the same.
        Duplicate parameters with the same name are not allowed.
        """
        if not isinstance(other, type(self)):
            raise ValueError("Can only merge configs of the same type")

        if not partial_equality(self, other, exclude=self._merge_exclude):
            raise ValueError(
                f"Can only merge configs that are equal except for {self._merge_exclude}"
            )

        merged = self.model_dump(by_alias=True, exclude=self._merge_exclude)
        for attr in self._merge_exclude:
            if merge_func := getattr(self, f"_merge_{attr}", None):
                merged[attr] = merge_func(other)
            elif isinstance(getattr(self, attr), list):
                self_attr = getattr(self, attr)
                merged[attr] = self_attr + [
                    x for x in getattr(other, attr) if x not in self_attr
                ]
            else:
                raise ValueError(
                    f"No merge protocol defined for {attr} in {type(self)}"
                )
        result = type(self)(**merged)
        if finalise:
            result.finalise()
        return result

    def finalise(self):
        # Check parameter names are unique
        seen = set()
        for param in self.parameters:
            assert param.name not in seen, "Parameter names should be unique"
            seen.add(param.name)

    def _format_out(self, param: ParamConfig, req: dict) -> dict:
        out = req.copy()
        out.pop("number", None)
        return out

    def in_mars(self, sources: Optional[list[str]] = None) -> Iterator:
        seen = set()
        for param in self.parameters:
            for req in param.in_keys(self.inputs, sources):
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
        for param, output in itertools.product(self.parameters, outputs):
            for req in param.out_keys(self.inputs):
                req["target"] = (
                    output.target.path
                    if hasattr(output.target, "path")
                    else output.target.type_
                )
                req.update(extract_mars(output.metadata))
                req.update(extract_mars(self.outputs.overrides))
                req = self._format_out(param, req)
                req.pop("interpolate", None)
                if req not in seen:
                    seen.append(req)
                    yield req

    @classmethod
    def _populate_param(
        cls,
        config: dict,
        input_config: list[dict],
        src_name: Optional[str] = None,
        nested: bool = False,
        **overrides,
    ):
        sort_inputs = cls._populate_inputs(
            input_config, [], **overrides.get("inputs", {})
        )
        if src_name is not None:
            reqs = sort_inputs[src_name]["request"]
            input_config = [reqs] if isinstance(reqs, dict) else reqs
        accums = cls._populate_accumulations(
            input_config, config.pop("accumulations", {})
        )
        param_config = {
            "accumulations": accums,
            **config,
        }
        updated_inputs = cls._populate_inputs(
            input_config, accums.keys(), **overrides.pop("inputs", {})
        )
        if not nested:
            param_config["inputs"] = updated_inputs
        deep_update(param_config, overrides)
        return param_config

    @classmethod
    def _populate_accumulations(cls, inputs: list[dict], base_accum: dict) -> dict:
        req = inputs[0]
        accums = base_accum.copy()

        # Populate coords in accumulations from inputs
        for dim, acc_config in accums.items():
            if dim == "step":
                # Handled separately below
                continue
            acc_config["coords"] = (
                [req[dim]]
                if acc_config.get("operation", None)
                else [[x] for x in req[dim]]
            )

        # Most entrypoints don't handle array with level dimension, so put this into accumulations to
        # separate different levels
        if levelist := req.get("levelist", None):
            levelist = [levelist] if np.ndim(levelist) == 0 else levelist
            accums.setdefault("levelist", {"coords": [[level] for level in levelist]})

        steps = req.get("step", None)
        if steps is not None:
            step_accum = accums.setdefault("step", {})
            if isinstance(steps, (int, str)):
                steps = [steps]

            if len(steps) > 2:
                diff = np.diff(steps)
                if len(set(diff)) == 1:
                    steps = {"from": steps[0], "to": steps[-1], "by": int(diff[0])}
            step_accum["coords"] = [steps]

            if step_accum.get("type") == "legacywindow":
                window_list = (
                    "std_anomaly_windows"
                    if step_accum.get("std_anomaly")
                    else "windows"
                )
                accums["step"] = {
                    "type": step_accum.pop("type"),
                    window_list: [step_accum],
                }
        return accums

    @classmethod
    def _populate_inputs(
        cls, inputs: list[dict], accum_dims: list[str], **overrides
    ) -> dict:
        [req.pop(dim, None) for req in inputs for dim in accum_dims]
        src_name = "fc"
        src_overrides = overrides.get(src_name, {})
        request_overrides = src_overrides.pop("request", {})
        updated_inputs = update_request(inputs, request_overrides)
        return {
            src_name: {
                "request": updated_inputs
                if len(updated_inputs) > 1
                else updated_inputs[0],
                **src_overrides,
            }
        }

    def _merge_parameters(self, other: Self) -> list[ParamConfig]:
        current_params = {
            cparam.name: cparam.model_dump(by_alias=True) for cparam in self.parameters
        }

        other_params = {
            oparam.name: oparam.model_dump(by_alias=True) for oparam in other.parameters
        }
        merged_params = []
        for name in list(current_params.keys()) + [
            x for x in other_params.keys() if x not in current_params
        ]:
            current_param = self.param(name)
            other_param = other.param(name)
            if current_param and other_param:
                merged_params.append(current_param.merge(other_param))
            else:
                merged_params.append(current_param or other_param)
        return merged_params
