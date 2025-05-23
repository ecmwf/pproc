# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import pandas as pd
import numpy as np
import logging

from pproc.config import types
from pproc.config.base import BaseConfig
from pproc.config.utils import expand, squeeze
from pproc.schema.schema import Schema


logging.getLogger("pproc").setLevel(os.environ.get("PPROC_LOG", "INFO").upper())
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class ConfigFactory:
    types = {
        "pproc-accumulate": types.AccumConfig,
        "pproc-ensms": types.EnsmsConfig,
        "pproc-monthly-stats": types.MonthlyStatsConfig,
    }

    @classmethod
    def from_dict(cls, entrypoint: str, **config) -> BaseConfig:
        if entrypoint not in cls.types:
            raise ValueError(
                f"Config generation current not supported for {entrypoint}"
            )
        return cls.types[entrypoint](**config)

    @classmethod
    def _from_schema(
        cls, entrypoint: str, schema_config: dict, **overrides
    ) -> BaseConfig:
        if entrypoint not in cls.types:
            raise ValueError(
                f"Config generation current not supported for {entrypoint}"
            )
        return cls.types[entrypoint].from_schema(schema_config, **overrides)

    @classmethod
    def from_outputs(
        cls, schema: Schema, output_requests: list[dict], **overrides
    ) -> BaseConfig:
        entrypoint = None
        config = None
        expanded = list(expand(output_requests))
        reqs = squeeze(expanded, ["levelist", "number", "quantile"])
        for req in reqs:
            schema_config = schema.config_from_output(req)
            req_entry = schema_config.pop("entrypoint")
            req_config = cls._from_schema(req_entry, schema_config, **overrides)

            if entrypoint is None:
                entrypoint = req_entry
                config = req_config
            else:
                if entrypoint != req_entry:
                    raise ValueError("All requests must have the same entrypoint")
                config = config.merge(req_config)
        assert (
            config is not None
        ), f"No config generated for requests: {output_requests}"
        return config

    @classmethod
    def from_inputs(
        cls,
        schema: Schema,
        entrypoint: str,
        input_requests: list[dict],
        **overrides,
    ) -> BaseConfig:
        config = None
        expanded = list(expand(input_requests))
        for schema_config in schema.config_from_input(expanded, entrypoint=entrypoint):
            if config is None:
                config = cls._from_schema(entrypoint, schema_config, **overrides)
            else:
                config = config.merge(
                    cls._from_schema(entrypoint, schema_config, **overrides)
                )
        assert config is not None, f"No config generated for requests: {input_requests}"
        return config
