# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import yaml
import copy
from typing import Callable, Any, Optional, Iterator
from typing_extensions import Self
import logging

from pproc.config.utils import deep_update

logger = logging.getLogger(__name__)

UpdateFunc = Callable[[dict, dict], dict]
FilterFunc = Callable[[dict, str], Any]
MatchFunc = Callable[[dict, Any, Any], bool]

DEFAULT_UPDATE: UpdateFunc = deep_update
DEFAULT_FILTER: FilterFunc = dict.__getitem__
DEFAULT_MATCH: MatchFunc = lambda _, value, expected: value == expected


class BaseSchema:
    custom_update: dict[str, UpdateFunc] = {}
    custom_filter: dict[str, FilterFunc] = {}
    custom_match: dict[str, MatchFunc] = {}

    def __init__(self, schema: dict):
        self.all_filters, self.schema = self.expand(schema)
        self.filters = self.all_filters.difference(set(self.custom_filter.keys()))

    @classmethod
    def from_file(cls, schema_path: str) -> Self:
        with open(schema_path, "r") as f:
            schema = yaml.safe_load(f)
        return cls(schema)

    @classmethod
    def expand(cls, schema: dict) -> tuple[set, dict]:
        expanded = {}
        filters = set()
        for key, value in schema.items():
            if cls.is_subschema(key):
                filters.add(key.split(":")[1])
                expanded.setdefault(key, {})
                for sub_keys, sub_values in value.items():
                    new_filters, sub_expanded = cls.expand(sub_values)
                    filters.update(new_filters)
                    for sub_key in sub_keys.split("/"):
                        expanded[key][sub_key] = sub_expanded
            else:
                expanded[key] = value
        return filters, expanded

    @classmethod
    def is_subschema(cls, key: str) -> bool:
        return "filter" in key

    @classmethod
    def subschema(cls, key: str, schema: dict, request: dict) -> dict:
        _, mars_key = key.split(":")
        filter_value = cls.custom_filter.get(mars_key, DEFAULT_FILTER)(
            request, mars_key
        )
        ret = schema.get(filter_value, schema.get("*", None))
        if ret is None:
            raise ValueError(
                f"Filter value {filter_value} not found in schema {schema}, and no default provided"
            )
        assert isinstance(ret, dict), f"Subschema must be a dictionary."
        return ret

    @classmethod
    def _traverse(cls, sub_schema: dict, request: dict, config: dict) -> dict:
        for key, value in sub_schema.items():
            if cls.is_subschema(key):
                cls._traverse(cls.subschema(key, value, request), request, config)
            else:
                # TODO: Remove copies?
                cls.custom_update.get(key, DEFAULT_UPDATE)(
                    config, {key: copy.deepcopy(value)}
                )
        return config

    def traverse(self, request: dict, config: Optional[dict] = None) -> dict:
        if len(set.intersection(set(request.keys()), self.filters)) < len(self.filters):
            raise ValueError(
                f"Request {request} does not contain all required filters {self.filters}"
            )
        return self._traverse(self.schema, request, config or {})

    @classmethod
    def _find_matching(
        cls,
        schema: dict,
        configs: list[dict],
        **matching,
    ) -> Iterator[dict]:
        for key, value in schema.items():
            if cls.is_subschema(key):
                filter_key = key.split(":")[1]
                new_configs = []
                value_matched = False
                for filter_value in value.keys():
                    if filter_value == "*":
                        continue
                    for fout in configs:
                        new_fout = copy.deepcopy(fout)
                        if new_fout["recon_req"].get(filter_key, None) == filter_value:
                            value_matched = True
                        if (
                            new_fout["recon_req"].setdefault(
                                filter_key, copy.deepcopy(filter_value)
                            )
                            != filter_value
                        ):
                            continue
                        new_configs.extend(
                            cls._find_matching(
                                schema[key][filter_value],
                                [new_fout],
                                **matching,
                            )
                        )
                if "*" in schema[key].keys() and not value_matched:
                    new_configs.extend(
                        cls._find_matching(schema[key]["*"], configs, **matching)
                    )
                configs = new_configs
            else:
                [
                    cls.custom_update.get(key, DEFAULT_UPDATE)(
                        cfg, {key: copy.deepcopy(value)}
                    )
                    for cfg in configs
                ]

        for cfg in configs:
            is_match = True
            for key, value in matching.items():
                if not cls.custom_match.get(key, DEFAULT_MATCH)(
                    cfg["recon_req"],
                    cfg.get(key, value),
                    copy.deepcopy(value),
                ):
                    is_match = False
                    break
            if is_match:
                logger.debug("Matched config: %s", cfg)
                yield cfg

    def reconstruct(
        self, output_template: Optional[dict] = None, **matching
    ) -> Iterator[tuple[dict, dict]]:
        for cfg in self._find_matching(
            self.schema, [{"recon_req": output_template or {}}], **matching
        ):
            yield cfg.pop("recon_req"), cfg
