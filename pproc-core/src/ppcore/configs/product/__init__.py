import logging
from typing import Optional, Union
from annotated_types import Annotated
import copy
from pydantic import Field

from ppcore.configs.product.ensemble import Ensemble

logger = logging.getLogger(__name__)

ENTRYPOINT_TO_CONFIG = {
    "pproc-ensms": "ensemble",
    "pproc-quantiles": "ensemble",
    "pproc-probabilities": "ensemble",
    "pproc-accumulate": "ensemble",
    "pproc-extreme": "extreme",
}


ProductConfig = Annotated[
    Union[Ensemble],
    Field(discriminator="name"),
]


def from_schema(
    schema: dict, request: dict, metadata: Optional[dict] = None, **overrides
) -> ProductConfig:
    name = schema.pop("config", "ensemble")
    schema.pop("name")
    schema.pop("entrypoint", None)
    interp_keys = schema.pop("interp_keys", {})
    dtype = schema.pop("dtype", "float32")
    inputs = copy.deepcopy(schema.pop("inputs"))
    accums = schema.pop("accumulations", None)

    for req in inputs:
        if grid := req.pop("target_grid", None):
            req["interpolate"] = {
                "grid": grid,
                **interp_keys,
            }

    config = {
        "name": name,
        "requests": inputs,
        "dtype": dtype,
        "sources": request.pop("sources", None),
        "outputs": request.pop("outputs", None),
        "metadata": metadata,
        **schema,
    }

    if accums is not None:
        # Populate coords in accumulations with values from inputs
        for dim, accum in accums.items():
            accum["coords"] = inputs[0][dim]
        config["accumulations"] = accums

    logger.info("Schema config", config)
    base = ProductConfig(**config)
    return ProductConfig(**{**base.model_dump(), **overrides})
