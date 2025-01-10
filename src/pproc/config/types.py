from typing import Optional, List
import numpy as np


from pproc.config.base import BaseConfig, Parallelisation
from pproc.config import io
from pproc.config.param import ParamConfig


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
    outputs: io.HistogramOutputModel = io.HistogramOutputModel()
    parameters: list[HistParamConfig]


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
        cls, schema: dict, output_requests: List[dict], **overrides
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
        cls, schema: dict, entrypoint: str, input_requests: List[dict], **overrides
    ) -> BaseConfig:
        config = None
        for req in input_requests:
            for schema_config in schema.config_from_input(entrypoint, req):
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
