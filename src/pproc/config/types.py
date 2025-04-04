import os
from typing import Optional, List, Any, Annotated, Iterator
from typing_extensions import Self, Union
from pydantic import model_validator, Field, Tag, Discriminator
import numpy as np
import datetime
import pandas as pd

from conflator import CLIArg


from pproc.config.base import BaseConfig, Parallelisation
from pproc.config import io
from pproc.config.param import ParamConfig
from pproc.config.utils import expand, _set, _get, update_request
from pproc.common.stepseq import steprange_to_fcmonth, fcmonth_to_steprange


class EnsmsConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.EnsmsOutputModel = io.EnsmsOutputModel()


class QuantilesConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.QuantilesOutputModel = io.QuantilesOutputModel()
    quantiles: int | List[float] = 100
    _total_number: int = 0
    _even_spacing: bool = None

    @property
    def even_spacing(self) -> bool:
        if self._even_spacing is None:
            self._even_spacing = isinstance(self.quantiles, int) or np.all(
                np.diff(self.quantiles) == self.quantiles[1] - self.quantiles[0]
            )
        return self._even_spacing

    @property
    def total_number(self) -> int:
        if self._total_number == 0:
            num_quantiles = (
                self.quantiles
                if isinstance(self.quantiles, int)
                else (len(self.quantiles) - 1)
            )
            self._total_number = num_quantiles if self._even_spacing else 100
        return self._total_number

    def quantile_indices(self, index: int) -> List[int]:
        pert_number = index if self.even_spacing else int(self.quantiles[index] * 100)
        return pert_number, self.total_number

    def _format_out(self, param: ParamConfig, req) -> dict:
        req = super()._format_out(param, req)
        num_quantiles = (
            self.quantiles
            if isinstance(self.quantiles, int)
            else (len(self.quantiles) - 1)
        )
        req["quantile"] = [
            f"{qindices[0]}:{qindices[1]}"
            for qindices in map(self.quantile_indices, range(num_quantiles + 1))
        ]
        return req


class AccumParamConfig(ParamConfig):
    vmin: Optional[float] = None
    vmax: Optional[float] = None


class AccumConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.AccumOutputModel = io.AccumOutputModel()
    parameters: list[AccumParamConfig]

    def _format_out(self, param: AccumParamConfig, req: dict) -> dict:
        req = req.copy()
        if req["type"] not in ["fcmean", "fcmax", "fcstdev", "fcmin"]:
            return req

        src_name = self.sources.names[0]
        source = param.in_sources(self.sources, src_name)
        src_reqs = source[0].request
        if isinstance(src_reqs, dict):
            src_reqs = [src_reqs]

        number = None
        num_members = super().compute_totalfields(src_name)
        for src_req in src_reqs:
            if len(src_req) == 0:
                continue
            number = src_req.get("number", number)

        if len(number) == num_members - 1:
            number = [0] + number
        req["number"] = number
        return req


class MonthlyStatsConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.MonthlyStatsOutputModel = io.MonthlyStatsOutputModel()
    parameters: list[AccumParamConfig]

    def _format_out(self, param: ParamConfig, req: dict) -> dict:
        req = req.copy()
        step_ranges = req.pop("step")
        date = datetime.datetime.strptime(str(req["date"]), "%Y%m%d")
        fcmonths = [
            steprange_to_fcmonth(date, step_range) for step_range in step_ranges
        ]
        req["fcmonth"] = fcmonths
        return req


class HistParamConfig(ParamConfig):
    bins: List[float]
    mod: Optional[int] = None
    normalise: bool = True
    scale_out: Optional[float] = None

    def _format_out(self, param: ParamConfig, req: dict) -> dict:
        req = super()._format_out(param, req)
        req["quantile"] = [f"{x}:{len(self.bins)}" for x in range(1, len(self.bins))]
        return req


class HistogramConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.HistogramOutputModel = io.HistogramOutputModel()
    parameters: list[HistParamConfig]


class ClimParamConfig(ParamConfig):
    clim: ParamConfig

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: Any) -> Any:
        clim = _get(data, "clim", {})
        if isinstance(clim, dict):
            clim_options = {**data, **clim}
            _set(data, "clim", ParamConfig(**clim_options))
        return data


class SigniParamConfig(ClimParamConfig):
    clim_em: ParamConfig
    epsilon: Optional[float] = None
    epsilon_is_abs: bool = True

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: Any) -> Any:
        clim = _get(data, "clim", {})
        if isinstance(clim, dict):
            clim_options = {**data, **clim}
            _set(data, "clim", ParamConfig(**clim_options))
        clim_em = _get(data, "clim_em", {})
        if isinstance(clim_em, dict):
            if len(clim_em) > 0:
                clim_options = {**data, **clim_em}
            else:
                clim_options = {**data, **clim}
            _set(data, "clim_em", ParamConfig(**clim_options))
        return data


class SigniConfig(BaseConfig):
    clim_total_fields: Annotated[int, Field(validate_default=True)] = 0
    parallelisation: Parallelisation = Parallelisation()
    sources: io.SignificanceSourceModel
    outputs: io.SignificanceOutputModel = io.SignificanceOutputModel()
    parameters: list[SigniParamConfig]
    use_clim_anomaly: Annotated[
        bool,
        CLIArg("--use-clim-anomaly", action="store_true", default=None),
        Field(description="Use anomaly of climatology in significance computation"),
    ] = False

    @model_validator(mode="after")
    def validate_totalfields(self) -> Self:
        super().validate_totalfields()
        if self.clim_total_fields == 0:
            self.clim_total_fields = self.total_fields("clim")
        return self


class AnomalyConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    sources: io.ClimSourceModel
    outputs: io.AnomalyOutputModel = io.AnomalyOutputModel()
    parameters: list[ClimParamConfig]


def anom_discriminator(config: Any) -> str:
    clim = _get(config, "clim", None)
    return "clim" if clim else "base"


class ProbParamConfig(ClimParamConfig):
    clim: Optional[ParamConfig] = None

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: Any) -> Any:
        clim = _get(data, "clim", None)
        if isinstance(clim, dict):
            clim_options = {
                **data,
                "preprocessing": [],
                "accumulations": {},
                "metadata": {},
                **clim,
            }
            _set(data, "clim", ParamConfig(**clim_options))
        return data


class ProbConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    sources: Annotated[
        Union[
            Annotated[io.BaseSourceModel, Tag("base")],
            Annotated[io.ClimSourceModel, Tag("clim")],
        ],
        Discriminator(anom_discriminator),
    ]
    outputs: io.ProbOutputModel = io.ProbOutputModel()
    parameters: list[ProbParamConfig]

    @model_validator(mode="after")
    def validate_param(self) -> Self:
        if isinstance(self.sources, io.ClimSourceModel):
            for param in self.parameters:
                if not param.clim:
                    param.clim = ParamConfig(
                        **param.model_dump(
                            exclude=("preprocessing", "accumulations", "metadata"),
                            by_alias=True,
                        ),
                        accumulations={},
                    )
        return self


class ExtremeParamConfig(ClimParamConfig):
    eps: float
    sot: list[int] = []

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: Any) -> Any:
        clim = _get(data, "clim", {})
        if isinstance(clim, dict):
            clim_options = {
                **data,
                "preprocessing": [],
                "accumulations": {},
                "metadata": {},
                **clim,
            }
            _set(data, "clim", ParamConfig(**clim_options))
        return data


class ExtremeConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    sources: io.ClimSourceModel
    outputs: io.ExtremeOutputModel = io.ExtremeOutputModel()
    parameters: list[ExtremeParamConfig]

    def _format_out(self, param: ParamConfig, req: dict) -> dict:
        req = super()._format_out(param, req)
        if req["type"] == "sot":
            req["number"] = param.sot
        return req


class WindParamConfig(ParamConfig):
    vod2uv: bool = False
    total_fields: int = 1

    @model_validator(mode="after")
    def set_vod2uv(self) -> Self:
        self.vod2uv = (
            self.sources.get("fc", {})
            .get("request", {})
            .get("interpolate")
            .get("vod2uv", False)
        )
        if self.vod2uv:
            self.total_fields = 2
        return self

    def in_sources(
        self, sources: io.SourceCollection, name: str, **kwargs
    ) -> list[io.Source]:
        base_config: io.Source = getattr(sources, name)
        config = self.sources.get(name, {})
        reqs = update_request(
            base_config.request,
            config.get("request", {}),
            **kwargs,
            **sources.overrides,
        )
        if self.vod2uv:
            return [
                io.Source(
                    type=config.get("type", base_config.type),
                    path=config.get("path", base_config.path),
                    request=reqs,
                )
            ]

        if isinstance(reqs, dict):
            reqs = expand(reqs, "param")
        else:
            reqs = sum([list(expand(req, "param")) for req in reqs], [])

        return [
            io.Source(
                type=config.get("type", base_config.type),
                path=config.get("path", base_config.path),
                request=items.to_dict("records"),
            )
            for _, items in pd.DataFrame(reqs).groupby("param")
        ]


class WindConfig(BaseConfig):
    parallelisation: int = 1
    outputs: io.WindOutputModel = io.WindOutputModel()
    parameters: list[WindParamConfig]

    def _format_out(self, param: WindParamConfig, req: dict) -> dict:
        req = req.copy()
        if req["type"] in ["em", "es"]:
            req.pop("number", None)
        return req


class ThermoParamConfig(ParamConfig):
    out_params: list[str]


class ThermoConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    sources: io.ThermoSourceModel
    outputs: io.ThermoOutputModel = io.ThermoOutputModel()
    parameters: list[ThermoParamConfig]
    validateutci: bool = False
    utci_misses: bool = False

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        for param in self.parameters:
            sources = param.in_sources(self.sources, "accum")
            if any([src.type == "null" for src in sources]):
                for out_req in param.accumulations["step"].out_mars("step"):
                    steps = out_req["step"]
                    if isinstance(steps, (str, int)):
                        steps = [steps]
                    nsteps = list(map(lambda x: len(str(x).split("-")), steps))
                    assert np.all(
                        np.asarray(nsteps) == 1
                    ), f"Accumulation source required for step ranges."
        return self
