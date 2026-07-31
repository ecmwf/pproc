# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional, Union

from earthkit.workflows.graph import Graph, deduplicate_nodes
from earthkit.workflows.fluent import Action
from earthkit.workflows.nodetree import combine_by_coords, datacubes
from qubed import Qube

from ppcore.utils.requests import expand
from ppcore.configs import from_outputs as config_from_outputs
from ppcore.configs.product import ProductConfig
from ppcore.schema.schema import Schema
from ppcore.schema.forecast import (
    ForecastDefinition,
    ReforecastDefinition,
    ClimatologyDefinition,
)
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


def product_from_output(
    request: dict,
    pproc_schema: Union[str, os.PathLike, Schema],
    forecast: Union[str, ForecastDefinition, ReforecastDefinition],
    climatology: Optional[Union[str, ClimatologyDefinition]] = None,
    metadata: Optional[dict] = None,
) -> Product:
    """
    Returns fluent.Action for computing the product specified by the output request
    and PProc schema
    """
    requests = list(expand(request))
    if len(requests) != 1:
        raise ValueError(
            f"Expected a single request after expansion, got {len(requests)}"
        )
    config = list(
        config_from_outputs(
            requests, pproc_schema, forecast, climatology, metadata=metadata
        )
    )[0]
    return product_from_config(config)


def action_from_outputs(
    requests: list[dict],
    pproc_schema: Union[str, os.PathLike, Schema],
    forecast: Union[str, Action, ForecastDefinition, ReforecastDefinition],
    climatology: Optional[Union[str, Action, ClimatologyDefinition]] = None,
    metadata: Optional[dict] = None,
) -> Action:
    action_kwargs = {}
    if isinstance(forecast, Action):
        action_kwargs["forecast"] = forecast
        dataset = ForecastDefinition(datacubes=datacubes(forecast.nodes))
    else:
        action_kwargs["forecast"] = None
        dataset = forecast
    if isinstance(climatology, Action):
        action_kwargs["climatology"] = climatology
        clim_dataset = ClimatologyDefinition(datacubes=datacubes(climatology.nodes))
    else:
        clim_dataset = climatology

    expanded_requests = list(expand(requests))

    # Determine coords that might be joined over and promote to dimension
    # TODO: enable generating outputs with a mix of instantaneous steps and step ranges
    # Should be done with changing more generally all values for "step" in requests to be
    # strings
    qube = Qube.empty()
    for request in expanded_requests:
        qube = qube | qube.from_datacube(request)
    promote = {dim for dim, values in qube.axes().items() if len(values) > 1}

    nodes = []
    for request in expanded_requests:
        nodes.append(
            product_from_output(
                request,
                pproc_schema=pproc_schema,
                forecast=dataset,
                climatology=clim_dataset,
                metadata=metadata,
            )
            .action(**action_kwargs, final_dims=promote)
            .nodes
        )
    return Action(combine_by_coords(nodes))


def graph_from_outputs(
    requests: list[dict],
    pproc_schema: Union[str, os.PathLike, Schema],
    forecast: Union[str, ForecastDefinition, ReforecastDefinition],
    climatology: Optional[Union[str, ClimatologyDefinition]] = None,
    metadata: Optional[dict] = None,
) -> Graph:
    """
    Returns fluent.Action for computing the product specified by the output request
    and PProc schema
    """
    graph = Graph([])
    for request in expand(requests):
        graph += (
            product_from_output(
                request,
                pproc_schema=pproc_schema,
                forecast=forecast,
                climatology=climatology,
                metadata=metadata,
            )
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
