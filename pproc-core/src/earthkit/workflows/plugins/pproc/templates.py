# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Optional

from earthkit.workflows.plugins.pproc.fluent import Action
from earthkit.workflows.plugins.pproc.config import schema_to_config
from ppcore.schema.schema import Schema


def derive_template(
    request: dict,
    pproc_schema: str,
    inputs: Optional[list[dict]] = None,
    metadata: Optional[dict] = None,
) -> Action:
    schema = Schema.from_file(pproc_schema)
    schema_config = schema.config_from_output(request, inputs=inputs)
    return schema_to_config(schema_config, request, metadata)


def from_request(
    request: dict,
    pproc_schema: str,
    preprocessing_dim: str = "param",
    ensemble_dim: str = "number",
    metadata: Optional[dict] = None,
    **sources: Action,
) -> Action:
    config = derive_template(request, pproc_schema, metadata=metadata)
    return config.action(
        **sources, preprocessing_dim=preprocessing_dim, ensemble_dim=ensemble_dim
    )
