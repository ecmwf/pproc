import logging
from typing import Any, Optional
import copy

from annotated_types import Annotated
from conflator import CLIArg, ConfigModel
from pydantic import BeforeValidator, ConfigDict, Field, create_model, model_validator
from typing_extensions import Self

from pproc import common


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


def deep_update(original: dict, update: Any) -> dict:
    for key, value in update.items():
        if isinstance(value, dict):
            original[key] = deep_update(original.get(key, {}), value)
        else:
            original[key] = value
    return original


class SourceConfig(ConfigModel):
    model_config = ConfigDict(populate_by_name=True)

    type_: str = Field(alias="type")
    request: dict | list[dict]
    path: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": "file", "path": data, "request": {}}
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


def create_source_model(entrypoint: str, sources: list[str]):
    return create_model(
        f"{entrypoint.capitalize()}Sources",
        _names=(list[str], sources),
        **{
            source: (
                SourceConfig,
                ...,
            )
            for source in sources
        },
        __base__=ConfigModel,
    )


def create_output_model(entrypoint: str, outputs: list[str]):
    return create_model(
        f"{entrypoint.capitalize()}Outputs",
        _names=(list[str], outputs),
        **{output: (OutputConfig, OutputConfig()) for output in outputs},
        __base__=ConfigModel,
    )


class MembersModel(ConfigModel):
    start: int
    end: int


SourceModel = create_source_model("base", ["ens"])
OutputModel = create_output_model("base", [])


class BaseConfig(ConfigModel):
    members: int | MembersModel
    total_fields: Annotated[int, Field(validate_default=True)] = 0
    n_par_read: int = 1
    n_par_compute: int = 1
    queue_size: int = 1
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
    recover: Annotated[
        bool,
        CLIArg("--recover", action="store_true", default=False),
    ]
    sources: SourceModel
    outputs: OutputModel = OutputModel()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for name in self.outputs._names:
            target = getattr(self.outputs, name).target
            if self.n_par_compute > 1:
                target.enable_parallel(common.parallel)
            if self.recover:
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

    @model_validator(mode="before")
    def set_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        for name in ["sources", "outputs"]:
            section = data.get(name, {})
            if not isinstance(section, dict):
                continue

            defaults = section.pop("default", {})
            for sub in section.keys():
                if isinstance(section[sub], dict):
                    section[sub] = deep_update(copy.deepcopy(defaults), section[sub])

        return data
