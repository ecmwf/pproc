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
from pproc.config.param import ParamConfig
from pproc.config.utils import deep_update, extract_mars, _get, _set


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
    sources: io.BaseSourceModel
    outputs: io.BaseOutputModel = io.BaseOutputModel()
    parameters: list[ParamConfig]
    _init: bool = False

    def print(self):
        print(yaml.dump(self.model_dump(by_alias=True)))

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
            source = param.in_sources(self.sources, src_name)
            reqs = source[0].request
            if isinstance(reqs, dict):
                reqs = [reqs]
            for req in reqs:
                if len(req) == 0:
                    continue
                if number := req.get("number", None):
                    total_fields += len(number)
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
            total_fields = self.compute_totalfields(self.sources.names[0])
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

    def merge(self, other: Self) -> Self:
        """
        Merge two configs, where all elements except for parameters must be the same.
        Duplicate parameters with the same name are not allowed.
        """
        if not isinstance(other, type(self)):
            raise ValueError("Can only merge configs of the same type")

        current = self.model_dump(by_alias=True)
        current_params = {
            cparam["name"]: cparam for cparam in current.pop("parameters")
        }
        other_model = other.model_dump(by_alias=True)
        other_params = {
            oparam["name"]: oparam for oparam in other_model.pop("parameters")
        }

        if current != other_model:
            raise ValueError(
                f"Configs must be the same except for parameters: {current} vs {other_model}"
            )

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
        return type(self)(**current, parameters=merged_params)

    @classmethod
    def from_schema(cls, schema_config: dict, **overrides) -> Self:
        schema_config = copy.deepcopy(schema_config)
        config = {
            "sources": {"default": {"type": "fdb"}},
            "outputs": {"default": {"target": {"type": "fdb"}}},
            "parameters": cls._construct_params(schema_config),
        }
        deep_update(config, overrides)
        return cls(**config)

    @classmethod
    def _populate_accumulations(cls, schema_config: dict):
        req = schema_config["inputs"][0]
        accums = schema_config.setdefault("accumulations", {})
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
                accums["step"] = {
                    "type": step_accum.pop("type"),
                    "windows": [step_accum],
                }

    @classmethod
    def _construct_params(cls, schema_config: dict) -> dict:
        inputs = schema_config["inputs"]
        for req in inputs:
            if grid := req.pop("interp_grid", None):
                req["interpolate"] = {
                    "grid": grid,
                    **schema_config["defs"].get("interp_keys", {}),
                }

        cls._populate_accumulations(schema_config)
        for dim in schema_config["accumulations"].keys():
            [req.pop(dim, None) for req in inputs]
        return {
            schema_config.get("name", str(inputs[0]["param"])): {
                "sources": {
                    "fc": {"request": inputs if len(inputs) > 1 else inputs[0]}
                },
                "metadata": schema_config.pop("metadata", {}),
                **schema_config,
            }
        }

    def in_mars(self, sources: Optional[list[str]] = None) -> Iterator:
        seen = set()
        for param, name in itertools.product(self.parameters, self.sources.names):
            for psource in param.in_sources(self.sources, name):
                if sources and psource.type not in sources:
                    continue
                reqs = (
                    psource.request
                    if isinstance(psource.request, list)
                    else [psource.request]
                )
                for req in reqs:
                    req["source"] = (
                        psource.path if psource.path is not None else psource.type
                    )
                    accum_updates = (
                        getattr(param, name).accumulations
                        if hasattr(param, name)
                        else {}
                    )
                    accumulations = deep_update(
                        param.accumulations.copy(), accum_updates
                    )
                    req.update(
                        {
                            key: accum.unique_coords()
                            for key, accum in accumulations.items()
                        }
                    )
                    req.pop("interpolate", None)
                    if str(req) not in seen:
                        seen.add(str(req))
                        yield req

    def _format_out(self, param: ParamConfig, req: dict) -> dict:
        out = req.copy()
        out.pop("number", None)
        return out

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
            for req in param.out_keys(self.sources):
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
