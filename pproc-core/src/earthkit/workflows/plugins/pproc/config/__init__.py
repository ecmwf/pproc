from typing import Optional
from ppcore.config.preprocessing import PreprocessingConfig
from .ensemble import EnsembleConfig
from .extreme import ExtremeConfig
from .anomaly import AnomalyConfig

ENTRYPOINT_TO_CONFIG = {
    "pproc-ensms": "ensemble",
    "pproc-quantiles": "ensemble",
    "pproc-probabilities": "ensemble",
    "pproc-accumulate": "ensemble",
    "pproc-extreme": "extreme",
}

CONFIGS = {
    "ensemble": EnsembleConfig,
    "extreme": ExtremeConfig,
    "anomaly": AnomalyConfig,
}


def _translate_accum_op(accum: dict) -> str:
    OPS = {
        "aggregation": None,
        "difference": "diff",
        "maximum": "max",
        "minimum": "min",
        "mean": "mean",
        "standard_deviation": "std",
        "sum": "add",
    }
    operation = accum.setdefault("operation", "aggregation")
    if operation not in OPS:
        raise ValueError(f"Accumulation operation {operation} not supported")
    return OPS[operation]


def schema_to_config(
    schema_config: dict, request: dict, metadata: Optional[dict] = None
) -> object:
    entrypoint = schema_config.pop("entrypoint")
    config_name = schema_config.pop("config", ENTRYPOINT_TO_CONFIG[entrypoint])
    ensemble_operation = {
        "em": "mean",
        "es": "std",
        "pb": "quantiles",
        "ep": "threshold_prob",
        "fcmean": None,
        "fcstdev": None,
        "fcmin": None,
        "fcmax": None,
        "efi": "efi",
        "efic": "efi",
        "sot": "sot",
        "cf": None,
        "pf": None,
    }
    schema_config.pop("interp_keys")
    schema_config.pop("name", None)
    schema_config.pop("dtype", None)
    inputs = schema_config.pop("inputs")

    # Populate coords in accumulations with values from inputs
    accums = schema_config.pop("accumulations", {})
    for dim, accum in accums.items():
        accum["operation"] = _translate_accum_op(accum)
        values = inputs[0][dim]
        accum["coords"] = [values] if isinstance(values, (str, int)) else values

    config = {
        "inputs": inputs,
        "preprocessing": PreprocessingConfig(
            actions=schema_config.pop("preprocessing", [])
        ),
        "accumulations": accums,
        "stats": {
            "operation": ensemble_operation[request["type"]],
            **schema_config,
        },
    }
    config["stats"].setdefault("metadata", {}).setdefault("type", request["type"])
    if metadata:
        config["stats"]["metadata"].update(metadata)
    if config_name not in CONFIGS:
        raise ValueError(
            f"Unknown config {config_name}: supported configs are {list(CONFIGS.keys())}"
        )
    return CONFIGS[config_name](**config)
