import copy
from typing import Any, ClassVar, Optional, Union

from annotated_types import Annotated
from conflator import CLIArg, ConfigModel
from pydantic import (BeforeValidator, ConfigDict, Discriminator, Field, Tag,
                      create_model, model_validator)

from pproc.config import utils
from pproc.config.targets import (FDBTarget, FileSetTarget, FileTarget,
                                  NullTarget, OverrideTargetWrapper)


class Source(ConfigModel):
    model_config = ConfigDict(populate_by_name=True)

    type_: str = Field(alias="type")
    request: dict | list[dict] = {}
    path: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": "fileset", "path": data}
        return data


class SourceCollection(ConfigModel):
    names: ClassVar[list[str]]
    overrides: Annotated[
        dict,
        BeforeValidator(utils.validate_overrides),
        CLIArg(
            "--override-input",
            action="append",
            default=[],
            metavar="KEY=VALUE,...",
        ),
        Field(
            default_factory=dict, description="Override input requests with these keys"
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def set_defaults(cls, data: Any) -> Any:
        defaults = data.get("default")
        if defaults is None:
            return data

        for sub in cls.names:
            subsec = data.get(sub, {})
            data[sub] = utils.deep_update(copy.deepcopy(defaults), subsec)
        return data


def target_discriminator(target: Any):
    if utils._get(target, "overrides", None):
        return "override"
    return utils._get(target, "type", "null")


class Output(ConfigModel):
    target: Annotated[
        Union[
            Annotated[NullTarget, Tag("null")],
            Annotated[FileTarget, Tag("file")],
            Annotated[FileSetTarget, Tag("fileset")],
            Annotated[FDBTarget, Tag("fdb")],
            Annotated[OverrideTargetWrapper, Tag("override")],
        ],
        Discriminator(target_discriminator),
        Field(default_factory=NullTarget),
    ]
    metadata: dict = {}


class OutputsCollection(ConfigModel):
    names: ClassVar[Union[list[str], dict[str, dict]]]
    default: Output = Output()
    overrides: Annotated[
        dict,
        BeforeValidator(utils.validate_overrides),
        CLIArg(
            "--override-output",
            action="append",
            default=[],
            metavar="KEY=VALUE,...",
        ),
        Field(default_factory=dict, description="Override outputs with these keys"),
    ]

    @model_validator(mode="before")
    @classmethod
    def set_overrides(cls, data: Any) -> Any:
        defaults = data.get("default", {})
        overrides = data.get("overrides", None)
        for sub in cls.names:
            subsec = data.get(sub, {})
            # Insert default metadata for each output type
            def_metadata = cls.names[sub] if isinstance(cls.names, dict) else {}
            metadata = {**def_metadata, **utils._get(subsec, "metadata", {})}
            # Set target from default, if specified
            target = utils._get(subsec, "target", utils._get(defaults, "target", {}))
            if overrides:
                target["overrides"] = overrides
            data[sub] = {"target": target, "metadata": metadata}
        return data


def create_source_model(entrypoint: str, sources: list[str]):
    return create_model(
        f"{entrypoint.capitalize()}Sources",
        names=(ClassVar[list[str]], sources),
        **{
            source: (
                Source,
                ...,
            )
            for source in sources
        },
        __base__=SourceCollection,
    )


def create_output_model(entrypoint: str, outputs: Union[list[str], dict[str, dict]]):
    field_definitions = {
        output: (
            Output,
            Output(metadata=(outputs[output] if isinstance(outputs, dict) else {})),
        )
        for output in outputs
    }
    return create_model(
        f"{entrypoint.capitalize()}Outputs",
        names=(ClassVar[Union[list[str], dict[str, dict]]], outputs),
        **field_definitions,
        __base__=OutputsCollection,
    )
