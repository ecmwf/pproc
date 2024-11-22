import os
from typing import Any, Optional

from annotated_types import Annotated
from conflator import CLIArg, ConfigModel
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from pproc.config import io
from pproc.config.log import LoggingConfig
from pproc.config.param import ParamConfig


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
    config: Optional[dict] = {}

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


SourceModel = io.create_source_model("base", ["ens"])
OutputModel = io.create_output_model("base", [])


class BaseConfig(ConfigModel):
    log: LoggingConfig = LoggingConfig()
    members: int | Members
    total_fields: Annotated[int, Field(validate_default=True)] = 0
    parallelisation: int | Parallelisation = 1
    recovery: Recovery = Recovery()
    sources: SourceModel
    outputs: OutputModel = OutputModel()
    parameters: list[ParamConfig]
    _init: bool = False

    @model_validator(mode="after")
    def _init_targets(self) -> Self:
        if self._init:
            return self

        self.recovery.config = self.model_dump(exclude_defaults=True)
        for name in self.outputs.names:
            target = getattr(self.outputs, name).target
            if self.parallelisation.n_par_compute > 1:
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
