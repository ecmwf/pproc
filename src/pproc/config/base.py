import os
from typing import Any

from annotated_types import Annotated
from conflator import CLIArg, ConfigModel
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from pproc.config import io
from pproc.config.log import LoggingConfig
from pproc.config.param import ParamConfig
from pproc.config.utils import deep_update


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

    def merge(self, other: Self) -> Self:
        """
        Merge two configs, where all elements except for parameters must be the same.
        Duplicate parameters with the same name are not allowed.
        """
        if not isinstance(other, type(self)):
            raise ValueError("Can only merge configs of the same type")

        current = self.model_dump(by_alias=True)
        current_params = {
            current["name"]: current for current in current.pop("parameters")
        }
        other = other.model_dump(by_alias=True)
        other_params = {other["name"]: other for other in other.pop("parameters")}

        if current != other:
            raise ValueError(
                f"Configs must be the same except for parameters: {current} vs {other}"
            )

        merged_params = []
        for name in list(current_params.keys()) + [
            x for x in other_params.keys() if x not in current_params
        ]:
            current_param = current_params.get(name, {})
            other_param = other_params.get(name, {})
            if current_param and other_param:
                merged_params.append(
                    ParamConfig(**current_param).merge(ParamConfig(**other_param))
                )
            else:
                merged_params.append(ParamConfig(**current_param, **other_param))

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
                    [
                        (
                            {"range": [x, x]}
                            if isinstance(x, int)
                            else {"range": [x["from"], x["to"], x["by"]]}
                        )
                        for x in coords
                    ],
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
