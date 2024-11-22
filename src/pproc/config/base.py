import copy
import os
from typing import Any, ClassVar, Optional

from annotated_types import Annotated
from conflator import CLIArg, ConfigModel
from pydantic import BeforeValidator, ConfigDict, Field, create_model, model_validator
from typing_extensions import Self

from pproc import common
from pproc.config import LoggingConfig


def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    return dict(map(lambda s: s.split("="), items))


def parse_var_strs(items):
    """
    Parse a list of comma-separated lists of key-value pairs and return a dictionary
    """
    return parse_vars(sum((s.split(",") for s in items if s), start=[]))


def _get(obj, attr, default=None):
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _set(obj, attr, value):
    if isinstance(obj, dict):
        obj[attr] = value
    else:
        setattr(obj, attr, value)


def deep_update(original: dict, update: Any) -> dict:
    for key, value in update.items():
        if isinstance(value, dict):
            _set(original, key, deep_update(_get(original, key), value))
        else:
            _set(original, key, value)
    return original


class SourceConfig(ConfigModel):
    model_config = ConfigDict(populate_by_name=True)

    type_: str = Field(alias="type")
    request: dict | list[dict]

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": data, "request": {}}
        return data


class BaseSource(ConfigModel):
    names: ClassVar[list[str]]

    @model_validator(mode="before")
    def set_defaults(cls, data: Any) -> Any:
        defaults = _get(data, "default")
        if defaults is None:
            return data

        for sub in cls.names:
            subsec = _get(data, sub, {})
            _set(data, sub, deep_update(copy.deepcopy(defaults), subsec))
        return data


class OutputConfig(ConfigModel):
    model_config = ConfigDict(populate_by_name=True)

    type_: str = Field(alias="type", default="null:")
    metadata: dict = {}
    override_output: Annotated[
        dict,
        BeforeValidator(lambda x: common.config.parse_var_strs(x)),
        CLIArg(
            "--override-output",
            action="append",
            default=[],
            metavar="KEY=VALUE,...",
        ),
        Field(description="Override outputs with these keys"),
    ] = {}
    _target: common.io.Target = None

    @property
    def target(self) -> common.io.Target:
        if self._target is None:
            self._target = common.io.target_from_location(
                self.type_, overrides=self.override_output
            )
        return self._target


class BaseOutput(ConfigModel):
    names: ClassVar[list[str]]
    default: OutputConfig = OutputConfig()

    @model_validator(mode="after")
    def set_defaults(self) -> Self:
        for name in self.names:
            out_config = getattr(self, name).model_dump(exclude_defaults=True)
            if "type_" not in out_config:
                getattr(self, name).type_ = self.default.type_
        return self


def create_source_model(entrypoint: str, sources: list[str]):
    return create_model(
        f"{entrypoint.capitalize()}Sources",
        names=(ClassVar[list[str]], sources),
        **{
            source: (
                SourceConfig,
                ...,
            )
            for source in sources
        },
        __base__=BaseSource,
    )


def create_output_model(entrypoint: str, outputs: list[str] | dict[str, dict]):
    field_definitions = {
        output: (
            OutputConfig,
            OutputConfig(
                metadata=(outputs[output] if isinstance(outputs, dict) else {})
            ),
        )
        for output in outputs
    }
    return create_model(
        f"{entrypoint.capitalize()}Outputs",
        names=(ClassVar[list[str]], outputs),
        **field_definitions,
        __base__=BaseOutput,
    )


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
    ] = True
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


SourceModel = create_source_model("base", ["ens"])
OutputModel = create_output_model("base", [])


class BaseConfig(ConfigModel):
    log: LoggingConfig = LoggingConfig()
    members: int | Members
    total_fields: Annotated[int, Field(validate_default=True)] = 0
    parallelisation: Parallelisation = Parallelisation()
    override_input: Annotated[
        dict,
        BeforeValidator(lambda x: common.config.parse_var_strs(x)),
        CLIArg(
            "--override-input",
            action="append",
            default=[],
            metavar="KEY=VALUE,...",
        ),
        Field(description="Override input requests with these keys"),
    ]
    recovery: Recovery = Recovery()
    sources: SourceModel
    outputs: OutputModel = OutputModel()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.recovery.config = self.model_dump(exclude_defaults=True)
        for name in self.outputs.names:
            target = getattr(self.outputs, name).target
            if self.parallelisation.n_par_compute > 1:
                target.enable_parallel(common.parallel)
            if self.recovery.from_checkpoint:
                target.enable_recovery()

    @model_validator(mode="after")
    def check_totalfields(self) -> Self:
        if self.total_fields == 0:
            self.total_fields = (
                self.members
                if isinstance(self.members, int)
                else self.members.end - self.members.start + 1
            )
        return self
