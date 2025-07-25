# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any, ClassVar, Optional, Union, Tuple
import re

from annotated_types import Annotated
from conflator import CLIArg, ConfigModel
from pydantic import (
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    create_model,
    model_validator,
)

from pproc.config import utils
from pproc.config.targets import (
    FDBTarget,
    FileSetTarget,
    FileTarget,
    NullTarget,
    OverrideTargetWrapper,
)
from pproc.extremes.indices import SUPPORTED_INDICES

_LOCATION_RE = re.compile("^([a-z](?:[a-z0-9+-.])*):(.*)$", re.I)


def split_location(
    loc: str, default: Optional[str] = None
) -> Tuple[Optional[str], str]:
    m = _LOCATION_RE.fullmatch(loc)
    if m is None:
        return (default, loc)
    return m.groups()


class Source(ConfigModel):
    model_config = ConfigDict(populate_by_name=True)

    type_: str = Field(alias="type")
    request: dict | list[dict] = {}
    path: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: Any) -> Any:
        if isinstance(data, str):
            type_, loc = split_location(data, default="file")
            config = {"type": type_}
            if loc:
                config["path"] = loc
            return config
        return data

    @property
    def type(self) -> str:
        return self.type_

    def location(self) -> str:
        return f"{self.type}:{self.path}" if self.type == "file" else f"{self.type}:req"

    def legacy_config(self) -> dict:
        cfg = {self.type: {"req": self.request}}
        if self.type == "fileset":
            cfg[self.type]["req"] = utils.update_request(
                cfg[self.type]["req"], {"location": self.path}
            )
        return cfg

    def base_request(self) -> dict:
        if isinstance(self.request, dict):
            return self.request
        return {
            k: v
            for k, v in self.request[0].items()
            if all(v == x[k] for x in self.request)
        }


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
        if not isinstance(data, dict) or "default" not in data:
            return data
        def_source = data["default"]
        for sub in cls.names:
            subsec = Source.validate_source(data.setdefault(sub, {}))
            data[sub] = utils.deep_update(def_source, subsec)
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

    @model_validator(mode="before")
    @classmethod
    def validate_target(cls, data: Any) -> Any:
        if isinstance(data, str):
            type_, loc = split_location(data, default="file")
            config = {"type": type_}
            if loc:
                config["path"] = loc
            return {"target": config}
        return data


class OutputsCollection(ConfigModel):
    names: ClassVar[list[str]]
    metadata_defaults: ClassVar[dict[str, dict]]
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
        defaults = utils._get(data, "default", {})
        overrides = utils._get(data, "overrides", None)
        for sub in cls.names:
            subsec = Output.validate_target(utils._get(data, sub, {}))
            # Insert default metadata for each output type
            def_metadata = cls.metadata_defaults.get(sub, {})
            metadata = {
                **def_metadata,
                **utils._get(defaults, "metadata", {}),
                **utils._get(subsec, "metadata", {}),
            }
            # Set target from default, if specified
            target = utils._get(subsec, "target", utils._get(defaults, "target", {}))
            if overrides:
                utils._set(target, "overrides", overrides)
            utils._set(data, sub, {"target": target, "metadata": metadata})
        return data


def create_source_model(
    name: str, sources: list[str], optional: list[str] = [], **kwargs
):
    field_definitions = {
        source: (
            Annotated[
                Source,
                CLIArg(f"--in-{source}"),
                Field(description=f"Input source for {source}"),
            ],
            ...,
        )
        for source in sources
    }
    for source in optional:
        field_definitions[source] = (
            Annotated[
                Source,
                CLIArg(f"--in-{source}"),
                Field(description=f"Input source for {source}. Default is null."),
            ],
            Source(type="null"),
        )
    return create_model(
        f"{name}SourceModel",
        names=(ClassVar[list[str]], sources + optional),
        **field_definitions,
        __base__=SourceCollection,
        **kwargs,
    )


def create_output_model(
    name: str, outputs: Union[list[str], dict[str, dict]], **kwargs
):
    field_definitions = {
        output: (
            Annotated[
                Output,
                CLIArg(f"--out-{output}"),
                Field(
                    description=f"Output target for {output}",
                    default=Output(
                        metadata=(outputs[output] if isinstance(outputs, dict) else {})
                    ),
                ),
            ],
            ...,
        )
        for output in outputs
    }
    names = outputs if isinstance(outputs, list) else list(outputs.keys())
    return create_model(
        f"{name}OutputModel",
        names=(ClassVar[list[str]], names),
        metadata_defaults=(
            ClassVar[dict[str, dict]],
            outputs if isinstance(outputs, dict) else {},
        ),
        **field_definitions,
        __base__=OutputsCollection,
        **kwargs,
    )


BaseSourceModel = create_source_model("Base", ["fc"])
BaseOutputModel = create_output_model("Base", [])
EnsmsOutputModel = create_output_model(
    "Ensms",
    {"mean": {"type": "em"}, "std": {"type": "es"}},
)
QuantilesOutputModel = create_output_model("Quantiles", {"quantiles": {"type": "pb"}})
AccumOutputModel = create_output_model("Accum", ["accum"])
MonthlyStatsOutputModel = create_output_model("MonthlyStats", ["stats"])
HistogramOutputModel = create_output_model("Histogram", {"histogram": {"type": "pd"}})
SignificanceSourceModel = create_source_model("Significance", ["fc", "clim", "clim_em"])
SignificanceOutputModel = create_output_model(
    "Significance", {"signi": {"type": "taem"}}
)
ClimSourceModel = create_source_model("Clim", ["fc", "clim"])
AnomalyOutputModel = create_output_model(
    "Anomaly", {"ens": {"type": "fcmean"}, "ensm": {"type": "taem"}}
)
ProbOutputModel = create_output_model("Prob", ["prob"])
ExtremeOutputModel = create_output_model("Extreme", SUPPORTED_INDICES)
WindOutputModel = create_output_model(
    "Wind",
    {"mean": {"type": "em"}, "std": {"type": "es"}, "ws": {}},
)
ThermoSourceModel = create_source_model("Thermo", ["inst"], optional=["accum"])
ThermoOutputModel = create_output_model("Thermo", ["indices", "accum", "intermediate"])
ECPointOutputModel = create_output_model("ECPoint", ["bs", "wt", "perc"])
