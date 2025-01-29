import os
from typing import Any, Iterator, Optional
import yaml

from annotated_types import Annotated
from conflator import CLIArg, ConfigModel
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from pproc.config import io
from pproc.config.log import LoggingConfig
from pproc.config.param import ParamConfig
from pproc.config.utils import deep_update, extract_mars, expand


class Members(ConfigModel):
    start: int
    end: int


class Parallelisation(ConfigModel):
    n_par_read: int = 1
    n_par_compute: int = 1
    queue_size: int = 1


class Recovery(ConfigModel):
    enable_checkpointing: bool = True
    from_checkpoint: Annotated[
        bool,
        CLIArg("--recover", action="store_true", default=False),
        Field(description="Recover from checkpoint"),
    ] = False
    root_dir: str = os.getcwd()

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
            if (
                isinstance(self.parallelisation, int)
                and self.parallelisation > 1
                or self.parallelisation.n_par_compute > 1
            ):
                target.enable_parallel()
            if self.recovery.from_checkpoint:
                target.enable_recovery()
        self._init = True
        return self

    @model_validator(mode="after")
    def check_totalfields(self) -> Self:
        if self.total_fields == 0:
            self.total_fields = (
                self.members
                if isinstance(self.members, int)
                else self.members.end - self.members.start + 1
            )
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
        request = schema_config.pop("request")
        if levelist := request.pop("levelist", None):
            schema_config["accumulations"]["levelist"] = {
                "coords": [[level] for level in levelist]
            }
        if schema_config["accumulations"]["step"].get("type", None) == "legacywindow":
            window_config = schema_config["accumulations"].pop("step")
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
            schema_config["accumulations"]["step"] = {
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
        for dim in schema_config["accumulations"].keys():
            request.pop(dim, None)
        parallelisation = schema_config.pop("parallelisation", None)
        config = {
            "members": schema_config.pop("members"),
            "total_fields": schema_config.pop("total_fields", 0),
            "sources": {"default": {"type": "fdb"}},
            "outputs": {"default": {"target": {"type": "fdb"}}},
            "parameters": {
                str(request["param"]): {
                    "sources": {"fc": {"request": request}},
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
        for param in self.parameters:
            for name in self.sources.names:
                base_source = getattr(self.sources, name)
                source = io.Source(
                    type=param.sources[name].get("type", base_source.type),
                    path=param.sources[name].get("path", base_source.path),
                    request=param.in_keys(
                        name, base_source.request, **self.sources.overrides
                    ),
                )
                if sources and source.type not in sources:
                    continue
                req = source.request
                req["source"] = source.path if source.path is not None else source.type
                accum_updates = (
                    getattr(param, name).accumulations if hasattr(param, name) else {}
                )
                accumulations = deep_update(param.accumulations.copy(), accum_updates)
                req.update(
                    {key: accum.unique_coords() for key, accum in accumulations.items()}
                )
                for tp_req in expand(req, "type"):
                    yield self._set_number(tp_req)

    def _set_number(self, req: dict) -> dict:
        if req.get("type", None) not in ["pf", "fcmean", "fcmax", "fcstdev", "fcmin"]:
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
        base_req = getattr(self.sources, "fc").request
        base_req.update(self.sources.overrides)
        for param in self.parameters:
            for name in self.outputs.names:
                if name == "default":
                    continue
                output = getattr(self.outputs, name)
                if output.target.type_ == "null":
                    continue
                if targets and output.target.type_ not in targets:
                    continue
                req = base_req.copy()
                req["target"] = (
                    output.target.path
                    if hasattr(output.target, "path")
                    else output.target.type_
                )
                for update in param.out_keys():
                    req.update(update)
                    req.update(extract_mars(output.metadata))
                    req.update(extract_mars(self.outputs.overrides))
                    for tp_req in expand(req, "type"):
                        yield self._set_number(tp_req)
