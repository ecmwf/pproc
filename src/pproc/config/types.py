from typing import Optional, List

from pproc.config.base import BaseConfig, Parallelisation
from pproc.config.io import create_output_model
from pproc.config.param import ParamConfig


EnsmsOutputModel = create_output_model(
    "ensms", {"mean": {"type": "em"}, "std": {"type": "es"}}
)


class EnsmsConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: EnsmsOutputModel = EnsmsOutputModel()


QuantilesOutputModel = create_output_model("quantiles", ["quantiles"])


class QuantilesConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: QuantilesOutputModel = QuantilesOutputModel()
    quantiles: int | List[float] = 100


class AccumParamConfig(ParamConfig):
    vmin: Optional[float] = None
    vmax: Optional[float] = None


AccumOutputModel = create_output_model("accumulate", ["accum"])


class AccumConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: AccumOutputModel = AccumOutputModel()
    parameters: list[AccumParamConfig]


MonthlyStatsOutputModel = create_output_model("monthly-stats", ["stats"])


class MonthlyStatsConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: MonthlyStatsOutputModel = MonthlyStatsOutputModel()
    parameters: list[AccumParamConfig]


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
