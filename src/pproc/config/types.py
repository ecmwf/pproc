import os
from typing import Optional, List, Any, Annotated, Iterator
from typing_extensions import Self, Union
from pydantic import model_validator, Field, Tag, Discriminator
import numpy as np
import datetime
import pandas as pd
import logging

from conflator import CLIArg


from pproc.config.base import BaseConfig, Parallelisation
from pproc.config import io
from pproc.config.param import ParamConfig
from pproc.config.schema import Schema
from pproc.config.utils import expand, squeeze, _set, _get, update_request
from pproc.common.stepseq import steprange_to_fcmonth

logging.getLogger("pproc").setLevel(os.environ.get("PPROC_LOG", "INFO").upper())
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


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

    def out_mars(self, targets: Optional[list[str]] = None) -> Iterator:
        num_quantiles = (
            self.quantiles
            if isinstance(self.quantiles, int)
            else (len(self.quantiles) - 1)
        )
        for req in super().out_mars(targets):
            yield {
                **req,
                "quantile": [
                    f"{qindices[0]}:{qindices[1]}"
                    for qindices in map(self.quantile_indices, range(num_quantiles + 1))
                ],
            }


class AccumParamConfig(ParamConfig):
    vmin: Optional[float] = None
    vmax: Optional[float] = None


class AccumConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.AccumOutputModel = io.AccumOutputModel()
    parameters: list[AccumParamConfig]


class MonthlyStatsConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.MonthlyStatsOutputModel = io.MonthlyStatsOutputModel()
    parameters: list[AccumParamConfig]

    def out_mars(self, targets: Optional[list[str]] = None) -> Iterator:
        for req in super().out_mars(targets):
            step_ranges = req.pop("step")
            date = datetime.datetime.strptime(str(req["date"]), "%Y%m%d")
            fcmonths = [
                steprange_to_fcmonth(date, step_range) for step_range in step_ranges
            ]
            yield {**req, "fcmonth": fcmonths}


class HistParamConfig(ParamConfig):
    bins: List[float]
    mod: Optional[int] = None
    normalise: bool = True
    scale_out: Optional[float] = None

    def out_mars(self, targets: Optional[list[str]] = None) -> Iterator:
        for req in super().out_mars(targets):
            yield {
                **req,
                "quantile": [f"{x}:{len(self.bins)}" for x in range(1, len(self.bins))],
            }


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
    clim_num_members: int = 11
    clim_total_fields: int = 0
    parallelisation: Parallelisation = Parallelisation()
    sources: io.SignificanceSourceModel
    outputs: io.SignificanceOutputModel = io.SignificanceOutputModel()
    parameters: list[SigniParamConfig]
    use_clim_anomaly: Annotated[
        bool,
        CLIArg("--use-clim-anomaly", action="store_true", default=False),
        Field(description="Use anomaly of climatology in significance computation"),
    ] = False

    @model_validator(mode="after")
    def check_totalfields(self) -> Self:
        super().check_totalfields()
        if self.clim_total_fields == 0:
            self.clim_total_fields = self.clim_num_members
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
    sot: list[int]

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


class ThermoParamConfig(ParamConfig):
    out_params: list[str]


class ThermoConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    sources: io.ThermoSourceModel
    outputs: io.ThermoOutputModel = io.ThermoOutputModel()
    parameters: list[ThermoParamConfig]
    validateutci: bool = False
    utci_misses: bool = False


class ConfigFactory:
    types = {
        "pproc-accumulate": AccumConfig,
        "pproc-ensms": EnsmsConfig,
        "pproc-monthly-stats": MonthlyStatsConfig,
    }

    @classmethod
    def from_dict(cls, entrypoint: str, **config) -> BaseConfig:
        if entrypoint not in cls.types:
            raise ValueError(
                f"Config generation current not supported for {entrypoint}"
            )
        return cls.types[entrypoint](**config)

    @classmethod
    def _from_schema_config(
        cls, entrypoint: str, schema_config: dict, **overrides
    ) -> BaseConfig:
        if entrypoint not in cls.types:
            raise ValueError(
                f"Config generation current not supported for {entrypoint}"
            )
        return cls.types[entrypoint].from_schema_config(schema_config, **overrides)

    @classmethod
    def from_outputs(
        cls, schema: Schema, output_requests: List[dict], **overrides
    ) -> BaseConfig:
        entrypoint = None
        config = None
        expanded = sum([list(expand(x)) for x in output_requests], [])
        reqs = squeeze(expanded, ["levelist", "step", "fcmonth", "number", "quantiles"])
        for req in reqs:
            schema_config = schema.config_from_output(req)

            if entrypoint is None:
                entrypoint = schema_config.pop("entrypoint")
                config = cls._from_schema_config(entrypoint, schema_config, **overrides)
            else:
                if entrypoint != schema_config.pop("entrypoint"):
                    raise ValueError("All requests must have the same entrypoint")
                config = config.merge(
                    cls._from_schema_config(entrypoint, schema_config, **overrides)
                )
        assert (
            config is not None
        ), f"No config generated for requests: {output_requests}"
        return config

    @classmethod
    def from_inputs(
        cls, schema: Schema, entrypoint: str, input_requests: List[dict], **overrides
    ) -> BaseConfig:
        config = None
        expanded_requests = sum([list(expand(x)) for x in input_requests], [])
        df = pd.DataFrame(expanded_requests)
        for _, param_combination in schema.combined_params():
            selection = df.loc[df["param"] == param_combination[0]]
            for paramid, row in selection.iterrows():
                req = row.dropna().to_dict()
                condition = np.logical_and.reduce(
                    [df[k] == v for k, v in req.items() if k != "param"]
                )
                meets_condition = df.loc[
                    condition & df["param"].isin(param_combination)
                ]
                if len(meets_condition) == len(param_combination):
                    logger.debug(
                        f"Add param {paramid}, computed from {param_combination}"
                    )
                    df.loc[len(df)] = {**req, "param": tuple(param_combination)}

        for _, pgroup in df.groupby("param", sort=False):
            # Squeeze certain back together into a single request
            squeeze_dims = ["step", "number", "levelist"]
            reqs = list(squeeze(pgroup.to_dict("records"), squeeze_dims))
            for schema_config in schema.config_from_input(reqs, entrypoint=entrypoint):
                if config is None:
                    config = cls._from_schema_config(
                        entrypoint, schema_config, **overrides
                    )
                else:
                    config = config.merge(
                        cls._from_schema_config(entrypoint, schema_config, **overrides)
                    )
        assert config is not None, f"No config generated for requests: {input_requests}"
        return config
