from typing import Optional, List, Any, Annotated
from typing_extensions import Self
from pydantic import model_validator, Field

from conflator import CLIArg


from pproc.config.base import BaseConfig, Parallelisation
from pproc.config import io
from pproc.config.param import ParamConfig
from pproc.config.schema import Schema


class EnsmsConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.EnsmsOutputModel = io.EnsmsOutputModel()


class QuantilesConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.QuantilesOutputModel = io.QuantilesOutputModel()
    quantiles: int | List[float] = 100


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


class HistParamConfig(ParamConfig):
    bins: List[float]
    mod: Optional[int] = None
    normalise: bool = True
    scale_out: Optional[float] = None


class HistogramConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.HistogramOutputModel = io.HistogramOutputModel()
    parameters: list[HistParamConfig]


class SigniParamConfig(ParamConfig):
    clim: ParamConfig
    clim_em: ParamConfig
    epsilon: Optional[float] = None
    epsilon_is_abs: bool = True

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: Any) -> Any:
        clim_options = data.copy()
        if "clim" in data:
            clim_options.update(clim_options.pop("clim"))
        data["clim"] = ParamConfig(**clim_options)
        clim_options = data.copy()
        if "clim_em" in data:
            clim_options.update(data.pop("clim_em"))
        elif "clim" in data:
            clim_options.update(data.pop("clim"))
        data["clim_em"] = ParamConfig(**clim_options)
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


class AnomalyParamConfig(ParamConfig):
    clim: ParamConfig

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: Any) -> Any:
        clim_options = data.copy()
        if "clim" in data:
            clim_options.update(clim_options.pop("clim"))
        data["clim"] = ParamConfig(**clim_options)
        return data


class AnomalyConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    sources: io.AnomalySourceModel
    outputs: io.AnomalyOutputModel = io.AnomalyOutputModel()
    parameters: list[AnomalyParamConfig]


class ConfigFactory:
    types = {
        "pproc-accumulate": AccumConfig,
        "pproc-ensms": EnsmsConfig,
        "pproc-monthly-stats": MonthlyStatsConfig,
    }

    @classmethod
    def from_schema_config(
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
        for req in output_requests:
            schema_config = schema.config_from_output(req)

            if entrypoint is None:
                entrypoint = schema_config.pop("entrypoint")
                config = cls.from_schema_config(entrypoint, schema_config, **overrides)
            else:
                if entrypoint != schema_config.pop("entrypoint"):
                    raise ValueError("All requests must have the same entrypoint")
                config = config.merge(
                    cls.from_schema_config(entrypoint, schema_config, **overrides)
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
        for req in input_requests:
            for schema_config in schema.config_from_input(req, entrypoint=entrypoint):
                if config is None:
                    config = cls.from_schema_config(
                        entrypoint, schema_config, **overrides
                    )
                else:
                    config = config.merge(
                        cls.from_schema_config(entrypoint, schema_config, **overrides)
                    )
        assert config is not None, f"No config generated for requests: {input_requests}"
        return config
