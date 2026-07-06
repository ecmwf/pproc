import os
from typing import Iterator, Optional, Union

from ppcore.schema.schema import Schema
from ppcore.configs.product import from_schema, ProductConfig
from ppcore.utils.requests import expand


def config_from_output(
    request: dict,
    pproc_schema: Union[str, os.PathLike],
    inputs: Optional[list[dict]] = None,
    metadata: Optional[dict] = None,
    **overrides,
) -> ProductConfig:
    schema = Schema.from_file(pproc_schema)
    if inputs is None:
        schema_config = schema.config_from_output(request)
    else:
        schema_configs = list(
            schema.config_from_input(input_requests=inputs, output_template=request)
        )
        if len(schema_configs) != 1:
            raise ValueError(
                f"Expected a single schema configuration from input requests, got {len(schema_configs)} for output request {request}"
            )
        schema_config = schema_configs[0]
    return from_schema(schema_config, request, metadata, **overrides)


def from_outputs(
    requests: list[dict],
    pproc_schema: Union[str, os.PathLike],
    inputs: Optional[list[dict]] = None,
    metadata: Optional[dict] = None,
    **overrides,
) -> Iterator[ProductConfig]:
    """
    Returns product configuration from output request and PProc schema
    """
    schema = Schema.from_file(pproc_schema)
    for request in expand(requests):
        if inputs is None:
            schema_config = schema.config_from_output(request)
        else:
            schema_configs = list(
                schema.config_from_input(input_requests=inputs, output_template=request)
            )
            if len(schema_configs) != 1:
                raise ValueError(
                    f"Expected a single schema configuration from input requests, got {len(schema_configs)} for output request {request}"
                )
            schema_config = schema_configs[0]
        yield from_schema(schema_config, request, metadata, **overrides)


def from_inputs(
    requests: list[dict],
    pproc_schema: Union[str, os.PathLike],
    inputs: Optional[list[dict]] = None,
    metadata: Optional[dict] = None,
    **overrides,
) -> Iterator[ProductConfig]:
    raise NotImplementedError(
        "Generation of configuration from input requests not yet implemented"
    )
