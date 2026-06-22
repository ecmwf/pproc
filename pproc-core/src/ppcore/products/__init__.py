from typing import Optional

from earthkit.workflows.graph import Graph, deduplicate_nodes

from ppcore.utils.requests import expand
from ppcore.configs import from_outputs as config_from_outputs
from ppcore.configs.product import ProductConfig
from ppcore.products.base import Product
from ppcore.products.ensemble import Ensemble


_PRODUCTS = {"ensemble": Ensemble}


def product_from_config(
    config: ProductConfig, input_overrides: dict = {}, output_overrides: dict = {}
) -> Product:
    if config.name not in _PRODUCTS:
        raise ValueError(f"Product {config.name} not supported")
    return _PRODUCTS[config.name](
        config=config,
        input_overrides=input_overrides,
        output_overrides=output_overrides,
    )


def product_from_outputs(
    requests: list[dict],
    pproc_schema: str,
    metadata: Optional[dict] = None,
) -> Product:
    """
    Returns fluent.Action for computing the product specified by the output request
    and PProc schema
    """
    requests = list(
        expand(requests, exclude=["levelist", "number", "quantile", "hdate"])
    )
    config = list(config_from_outputs(requests, pproc_schema, metadata=metadata))[0]
    return product_from_config(config)


def graph_from_outputs(
    requests: list[dict],
    pproc_schema: str,
    metadata: Optional[dict] = None,
) -> Product:
    """
    Returns fluent.Action for computing the product specified by the output request
    and PProc schema
    """
    graph = (
        product_from_outputs(requests, pproc_schema=pproc_schema, metadata=metadata)
        .action()
        .graph()
    )
    return deduplicate_nodes(graph)


def graph_from_configs(
    configs: list[ProductConfig],
    input_overrides: dict = {},
    output_overrides: dict = {},
) -> Graph:
    graph = Graph([])
    for config in configs:
        graph += (
            product_from_config(config, input_overrides, output_overrides)
            .action()
            .graph()
        )
    return deduplicate_nodes(graph)
