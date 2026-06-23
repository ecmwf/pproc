from typing import Iterator, Optional

from ppcore.schema.schema import Schema
from ppcore.configs.product import from_schema, ProductConfig
from ppcore.utils.requests import expand


def config_from_output(
    request: dict,
    pproc_schema: str,
    inputs: Optional[list[dict]] = None,
    metadata: Optional[dict] = None,
    **overrides,
) -> ProductConfig:
    schema = Schema.from_file(pproc_schema)
    schema_config = schema.config_from_output(request, inputs=inputs)
    return from_schema(schema_config, request, metadata, **overrides)


def from_outputs(
    requests: list[dict],
    pproc_schema: str,
    inputs: Optional[list[dict]] = None,
    metadata: Optional[dict] = None,
    **overrides,
) -> Iterator[ProductConfig]:
    """
    Returns product configuration from output request and PProc schema
    """
    schema = Schema.from_file(pproc_schema)
    for request in expand(requests):
        schema_config = schema.config_from_output(request, inputs=inputs)
        yield from_schema(schema_config, request, metadata, **overrides)


def from_inputs(
    requests: list[dict],
    pproc_schema: str,
    inputs: Optional[list[dict]] = None,
    metadata: Optional[dict] = None,
    **overrides,
) -> Iterator[ProductConfig]:
    raise NotImplementedError(
        "Generation of configuration from input requests not yet implemented"
    )
