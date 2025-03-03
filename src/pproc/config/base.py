from typing import Any, Iterator, Optional
import yaml
import itertools
import numpy as np

from annotated_types import Annotated
from conflator import CLIArg, ConfigModel
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from pproc.config import io
from pproc.config.log import LoggingConfig
from pproc.config.param import ParamConfig
from pproc.config.utils import deep_update, extract_mars, _get, _set


class Members(ConfigModel):
    start: int
    end: int


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
    members: int | Members
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

    @model_validator(mode="after")
    def check_totalfields(self) -> Self:
        fc_name = self.sources.names[0]
        if self.total_fields == 0 and len(self.parameters) > 0:
            source = self.parameters[0].in_sources(self.sources, fc_name)
            if len(source) == 0:
                return self
            reqs = source[0].request
            if isinstance(reqs, dict):
                reqs = [reqs]
            for req in reqs:
                if number := req.get("number", None):
                    self.total_fields += 1 if isinstance(number, int) else len(number)
                elif req.get("type", None) in ["pf", "fcmean"]:
                    self.total_fields += (
                        self.members
                        if isinstance(self.members, int)
                        else self.members.end - self.members.start + 1
                    )
                else:
                    self.total_fields += 1
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
    def from_schema_config(cls, schema_config: dict, **overrides) -> Self:
        reqs = schema_config.pop("request")
        if not isinstance(reqs, list):
            reqs = [reqs]
        accum = schema_config.setdefault("accumulations", {})
        for req in reqs:
            if levelist := req.pop("levelist", None):
                levelist = [levelist] if np.ndim(levelist) == 0 else levelist
                accum.setdefault(
                    "levelist", {"coords": [[level] for level in levelist]}
                )
        if accum["step"].get("type", None) == "legacywindow":
            window_config = accum.pop("step")
            coords = window_config.pop("coords")
            if isinstance(coords, list):
                coords = [
                    (
                        {"range": [int(x["from"]), int(x["to"]), int(x["by"])]}
                        if isinstance(x, dict)
                        else {"range": [int(x[0]), int(x[0])]}
                    )
                    for x in coords
                ]
            accum["step"] = {
                "type": window_config.pop("type"),
                "windows": [
                    {
                        "window_operation": window_config.pop("operation", "none"),
                        "grib_set": window_config.pop("grib_keys", {}),
                        "periods": coords,
                        **window_config,
                    }
                ],
            }
        for dim in accum.keys():
            [req.pop(dim, None) for req in reqs]
        parallelisation = schema_config.pop("parallelisation", None)
        config = {
            "members": schema_config.pop("members"),
            "total_fields": schema_config.pop("total_fields", 0),
            "sources": {"fc": {"type": "fdb"}},
            "outputs": {"default": {"target": {"type": "fdb"}}},
            "parameters": {
                schema_config.get("name", str(reqs[0]["param"])): {
                    "sources": {"fc": {"request": reqs if len(reqs) > 1 else reqs[0]}},
                    "metadata": schema_config.pop("metadata", {}),
                    **schema_config,
                }
            },
        }
        if parallelisation:
            config["parallelisation"] = parallelisation
        deep_update(config, overrides)
        return cls(**config)

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
                    req = self._set_number(req)
                    if str(req) not in seen:
                        seen.add(str(req))
                        yield req

    def _set_number(self, req: dict) -> dict:
        if req.get("type", None) not in ["pf", "fcmean", "fcmax", "fcstdev", "fcmin"]:
            return req
        if "number" in req:
            return req
        if isinstance(self.members, int):
            if req["type"] == "pf":
                start, end = 1, self.members
            else:
                start, end = 0, self.members - 1
        else:
            start = self.members.start
            end = self.members.end
        return {**req, "number": list(range(start, end + 1))}

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

        seen = set()
        for param, output in itertools.product(self.parameters, outputs):
            for req in param.out_keys(self.sources):
                req["target"] = (
                    output.target.path
                    if hasattr(output.target, "path")
                    else output.target.type_
                )
                req.update(extract_mars(output.metadata))
                req.update(extract_mars(self.outputs.overrides))
                req = self._set_number(req)
                req.pop("interpolate", None)
                if str(req) not in seen:
                    seen.add(str(req))
                    yield req
