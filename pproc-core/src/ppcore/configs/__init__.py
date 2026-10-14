# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Iterator, Optional, Union

from ppcore.schema.schema import Schema
from ppcore.schema.forecast import (
    ForecastDefinition,
    ReforecastDefinition,
    ClimatologyDefinition,
)
from ppcore.configs.product import from_schema, ProductConfig
from ppcore.utils.requests import expand


def config_from_output(
    request: dict,
    pproc_schema: Union[str, os.PathLike, Schema],
    forecast: Union[ForecastDefinition, ReforecastDefinition],
    climatology: Optional[ClimatologyDefinition] = None,
    metadata: Optional[dict] = None,
    **overrides,
) -> ProductConfig:
    if isinstance(pproc_schema, Schema):
        schema = pproc_schema
    else:
        schema = Schema.from_file(pproc_schema)
    schema_config = schema.config_from_output(request, forecast, climatology)
    return from_schema(schema_config, request, metadata, **overrides)


def from_outputs(
    requests: list[dict],
    pproc_schema: Union[str, os.PathLike, Schema],
    forecast: Union[str, ForecastDefinition, ReforecastDefinition],
    climatology: Optional[Union[str, ClimatologyDefinition]] = None,
    metadata: Optional[dict] = None,
    **overrides,
) -> Iterator[ProductConfig]:
    """
    Returns product configuration from output request and PProc schema
    """
    if isinstance(pproc_schema, Schema):
        schema = pproc_schema
    else:
        schema = Schema.from_file(pproc_schema)
    if isinstance(forecast, str):
        forecast = schema.datasets.definition(forecast)
    if isinstance(climatology, str):
        climatology = schema.datasets.definition(climatology)
    for request in expand(requests):
        schema_config = schema.config_from_output(request, forecast, climatology)
        yield from_schema(schema_config, request, metadata, **overrides)


def from_inputs(
    template: dict,
    pproc_schema: Union[str, os.PathLike, Schema],
    forecast: Union[str, ForecastDefinition, ReforecastDefinition],
    climatology: Optional[Union[str, ClimatologyDefinition]] = None,
    metadata: Optional[dict] = None,
    **overrides,
) -> Iterator[ProductConfig]:
    """
    Returns product configuration from input forecast, template for the output request and PProc schema
    """
    if isinstance(pproc_schema, Schema):
        schema = pproc_schema
    else:
        schema = Schema.from_file(pproc_schema)
    if isinstance(forecast, str):
        forecast = schema.datasets.definition(forecast)
    if isinstance(climatology, str):
        climatology = schema.datasets.definition(climatology)
    for schema_config in schema.config_from_input(
        forecast, climatology, output_template=template
    ):
        yield from_schema(schema_config, template, metadata, **overrides)
