# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Annotated, Union, Literal, Self
from dataclasses import dataclass
from functools import cached_property
from qubed import Qube
from pydantic import Field, model_validator, field_validator
import bisect
import logging

from earthkit.workflows.plugins.pproc.utils.pydantic_utils import PProcBaseModel
from ppcore.utils.requests import validate_request

logger = logging.getLogger(__name__)


class BaseDefinition(PProcBaseModel):
    datacubes: list[dict]
    wave: Optional[list[dict]] = None


class ForecastDefinition(BaseDefinition):
    dataset_type: Literal["forecast"] = "forecast"
    unperturbed: Optional[dict] = None

    @field_validator("datacubes", mode="before")
    def populate_request(cls, data: list[dict]) -> list[dict]:
        data = data.copy()
        for index, cube in enumerate(data):
            if members := cube.pop("members", None):
                cube["number"] = list(range(members["start"], members["end"] + 1))
            steps = []
            cube_step = cube.pop("step", [])
            if isinstance(cube_step, (int, str)):
                cube_step = [cube_step]
            for step in cube_step:
                if isinstance(step, dict):
                    steps.extend(range(step["from"], step["to"] + 1, step.get("by", 1)))
                else:
                    steps.append(step)
            cube["step"] = steps
            if quantile := cube.pop("quantile", None):
                cube["quantile"] = [f"{x}:{quantile}" for x in range(0, quantile + 1)]
            data[index] = validate_request(cube)
        return data

    @model_validator(mode="after")
    def validate_qube(self) -> Self:
        if self.wave is None and any(["param" not in cube for cube in self.datacubes]):
            raise ValueError(
                "Forecast must contain a list of parameters or a wave configuration"
            )
        return self


class ReforecastDefinition(ForecastDefinition):
    dataset_type: Literal["reforecast"] = "reforecast"
    scheme: str


class ClimatologyDefinition(BaseDefinition):
    dataset_type: Literal["climatology"] = "climatology"
    scheme: str


DatasetDefinition = Annotated[
    Union[ForecastDefinition, ReforecastDefinition, ClimatologyDefinition],
    Field(discriminator="dataset_type"),
]


@dataclass(kw_only=True)
class Dataset:
    datacubes: list[dict]
    wave: Optional[list[dict]] = None

    @cached_property
    def qube(self) -> Qube:
        dataqube = Qube.empty()
        for cube in self.datacubes:
            dataqube = dataqube | Qube.from_datacube(cube)
        return dataqube

    @cached_property
    def wave_qube(self) -> Qube:
        if self.wave is None:
            raise ValueError("No wave configuration defined for this forecast")
        wave_qube = Qube.empty()
        for condition in self.wave:
            wave_qube = wave_qube | self.qube.select(condition)
        return wave_qube

    @cached_property
    def atmos_qube(self) -> Qube:
        if self.wave is None:
            raise ValueError("No wave configuration defined for this forecast")
        return self.qube - self.wave_qube

    def stream(self, request: dict) -> str:
        if self.wave is None:
            raise ValueError("No wave configuration defined for this forecast")
        for condition in self.wave:
            if all([request[key] == value for key, value in condition.items()]):
                return "wave"
        return "atmos"

    def stream_qube(self, stream: Optional[str]) -> Qube:
        if stream is None:
            return self.qube
        if stream == "wave":
            return self.wave_qube
        if stream == "atmos":
            return self.atmos_qube
        return self.qube.select({"stream": stream})

    def type_qube(self, tp: Optional[str], number: Optional[str]) -> Qube:
        qube = self.qube if tp is None else self.qube.select({"type": tp})
        if number is None or number == "{number}" or "number" not in qube.axes():
            return qube

        if isinstance(number, int):
            number = [number]
        number_axes = sorted(qube.axes()["number"])
        start = bisect.bisect_left(number, number_axes[0])
        end = bisect.bisect_right(number, number_axes[-1])
        return qube.select({"number": number[start:end]})

    def select(
        self, selection: dict, select_to_override: bool = False
    ) -> tuple[Qube, dict]:
        stream = selection.pop("stream", None)
        tp = selection.pop("type", None)
        number = selection.pop("number", None)
        stream_qube = self.stream_qube(stream)
        type_qube = self.type_qube(tp, number)
        qube = stream_qube & type_qube
        overrides = {}
        for key, vals in selection.items():
            logger.debug(f"Selecting {key}:{vals} from {qube}")
            if key in qube.axes():
                qube = qube.select({key: vals})
            elif select_to_override:
                overrides[key] = vals
            else:
                return Qube.empty(), {}
        return qube, overrides

    def steps(self, request: dict) -> list[int] | list[str]:
        time = request.get("time", None)
        qube = self.qube
        if time is not None and len(qube.axes().get("time", [])) > 1:
            qube = self.qube.select({"time": time})
            if qube.n_leaves == 0:
                raise ValueError(f"No datacubes available for time {time}")
        steps = list(qube.axes()["step"])
        return sorted(
            steps, key=lambda x: int(x.split("-")[0]) if isinstance(x, str) else x
        )

    def sample_datacube(self, qube: Optional[Qube] = None) -> dict:
        qube = qube or self.qube
        if qube.n_leaves == 0:
            raise ValueError("No datacubes available for this forecast")
        return validate_request(list(qube.datacubes())[0])


@dataclass(kw_only=True)
class Forecast(Dataset):
    fc_type: Literal["forecast"] = "forecast"
    unperturbed: Optional[dict] = None

    @cached_property
    def unperturbed_qube(self) -> Qube:
        if self.unperturbed is None:
            return Qube.empty()
        return self.qube.select(self.unperturbed)

    @cached_property
    def perturbed_qube(self) -> Qube:
        return self.qube - self.unperturbed_qube

    @cached_property
    def is_ensemble(self) -> bool:
        n_perturbed = len(self.qube.axes().get("number", []))
        n_unperturbed = 0
        if self.unperturbed is not None:
            n_unperturbed = 1 if self.type_qube("unperturbed").n_leaves > 0 else 0
        return (n_perturbed + n_unperturbed) > 1

    def type_qube(self, tp: Optional[str], number: Optional[str] = None) -> Qube:
        if tp is None:
            qube = self.qube
        elif tp == "unperturbed":
            if self.unperturbed is None:
                raise ValueError(
                    "No unperturbed configuration defined for this forecast"
                )
            qube = self.unperturbed_qube
        elif tp == "perturbed":
            if self.unperturbed is None:
                raise ValueError(
                    "No unperturbed configuration defined for this forecast"
                )
            qube = self.perturbed_qube
        else:
            qube = self.qube.select({"type": tp})

        if number is None or number == "{number}" or "number" not in qube.axes():
            return qube

        if isinstance(number, int):
            number = [number]
        number_axes = sorted(qube.axes()["number"])
        start = bisect.bisect_left(number, number_axes[0])
        end = bisect.bisect_right(number, number_axes[-1])
        return qube.select({"number": number[start:end]})

    def select(
        self, selection: dict, select_to_override: bool = False
    ) -> tuple[Qube, dict]:
        ensemble = selection.pop("ensemble", None)
        if ensemble is not None and ensemble != self.is_ensemble:
            return Qube.empty(), {}
        return super().select(selection, select_to_override=select_to_override)

    def sample_datacube(self, qube: Optional[Qube] = None, pop: list[str] = []) -> dict:
        qube = qube or self.qube
        if qube.n_leaves == 0:
            raise ValueError("No datacubes available for this forecast")
        if self.is_ensemble and "number" in qube.axes():
            number_axes = sorted(map(int, qube.axes()["number"]))
            qube = qube.select({"number": [str(number_axes[0])]})
        out = validate_request(list(qube.datacubes())[0])
        for key in pop:
            out.pop(key, None)
        return out


@dataclass(kw_only=True)
class Reforecast(Forecast):
    scheme: str


@dataclass(kw_only=True)
class Climatology(Dataset):
    scheme: str


class DatasetDefinitions(PProcBaseModel):
    definitions: dict[str, DatasetDefinition]

    def definition(self, name: str) -> DatasetDefinition:
        if name not in self.definitions:
            raise ValueError(f"Dataset {name} not defined")
        return self.definitions[name]


def definition_to_dataset(definition: DatasetDefinition) -> Dataset:
    if isinstance(definition, ForecastDefinition):
        return Forecast(
            datacubes=definition.datacubes,
            wave=definition.wave,
            unperturbed=definition.unperturbed,
        )
    elif isinstance(definition, ReforecastDefinition):
        return Reforecast(
            datacubes=definition.datacubes,
            wave=definition.wave,
            unperturbed=definition.unperturbed,
            scheme=definition.scheme,
        )
    elif isinstance(definition, ClimatologyDefinition):
        return Climatology(
            datacubes=definition.datacubes,
            wave=definition.wave,
            scheme=definition.scheme,
        )
    else:
        raise ValueError(f"Unknown dataset definition type: {type(definition)}")
