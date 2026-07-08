# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import copy
import logging
from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Optional

import yaml
from typing_extensions import Self

from ppcore.utils.dicts import deep_update

logger = logging.getLogger(__name__)

UpdateFunc = Callable[[dict, dict], dict]
FilterFunc = Callable[[dict, str], Any]
MatchFunc = Callable[[dict, Any, Any], bool]


def dict_update(base: dict, update: dict) -> dict:
    base.update(update)
    return base


DEFAULT_UPDATE: UpdateFunc = deep_update
DEFAULT_FILTER: FilterFunc = dict.__getitem__
DEFAULT_MATCH: MatchFunc = lambda _, value, expected: value == expected


class BaseSchema:
    custom_update: dict[str, UpdateFunc] = {}
    custom_filter: dict[str, FilterFunc] = {}
    custom_match: dict[str, MatchFunc] = {}

    def __init__(
        self,
        schema: dict,
        *,
        matching_cache_size: int = 0,
    ):
        self.all_filters, self.schema = self.expand(schema)
        self.filters = self.all_filters.difference(set(self.custom_filter.keys()))
        self.matching_cache_size = max(0, matching_cache_size)
        self._matching_cache: OrderedDict[tuple, tuple[dict, ...]] = OrderedDict()

    @classmethod
    def _freeze_cache_value(cls, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (key, cls._freeze_cache_value(item)) for key, item in value.items()
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze_cache_value(item) for item in value)
        if isinstance(value, set):
            frozen = [cls._freeze_cache_value(item) for item in value]
            return ("__set__", tuple(sorted(frozen, key=repr)))

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return (
                "__model__",
                cls._freeze_cache_value(model_dump(exclude_none=False, by_alias=True)),
            )

        return ("__repr__", repr(value))

    def _matching_cache_key(self, output_template: dict, matching: dict) -> tuple:
        return (
            id(self.schema),
            self._freeze_cache_value(output_template),
            self._freeze_cache_value(matching),
        )

    def clear_matching_cache(self) -> None:
        self._matching_cache.clear()

    def _get_cached_matching(self, key: tuple) -> Optional[list[dict]]:
        if self.matching_cache_size == 0:
            return None

        cached = self._matching_cache.get(key, None)
        if cached is None:
            return None

        self._matching_cache.move_to_end(key)
        return [copy.deepcopy(cfg) for cfg in cached]

    def _set_cached_matching(self, key: tuple, configs: list[dict]) -> None:
        if self.matching_cache_size == 0:
            return

        self._matching_cache[key] = tuple(copy.deepcopy(cfg) for cfg in configs)
        self._matching_cache.move_to_end(key)
        while len(self._matching_cache) > self.matching_cache_size:
            self._matching_cache.popitem(last=False)

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
        assert isinstance(ret, dict), "Subschema must be a dictionary."
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
        missing = object()
        matching_funcs = [
            (key, value, cls.custom_match.get(key, DEFAULT_MATCH))
            for key, value in matching.items()
        ]
        for key, value in schema.items():
            if cls.is_subschema(key):
                filter_key = key.split(":")[1]
                wildcard_schema = value.get("*", None)
                new_configs = []
                value_matched = False
                for filter_value, sub_schema in value.items():
                    if filter_value == "*":
                        continue
                    for fout in configs:
                        current_value = fout["recon_req"].get(filter_key, missing)
                        if current_value == filter_value:
                            value_matched = True
                        elif current_value is not missing:
                            continue

                        new_fout = copy.deepcopy(fout)
                        if current_value is missing:
                            new_fout["recon_req"][filter_key] = copy.deepcopy(
                                filter_value
                            )

                        new_configs.extend(
                            cls._find_matching(
                                sub_schema,
                                [new_fout],
                                **matching,
                            )
                        )
                if wildcard_schema is not None and not value_matched:
                    new_configs.extend(
                        cls._find_matching(wildcard_schema, configs, **matching)
                    )
                configs = new_configs
            else:
                update = cls.custom_update.get(key, DEFAULT_UPDATE)
                for cfg in configs:
                    update(cfg, {key: copy.deepcopy(value)})

        for cfg in configs:
            is_match = True
            for key, value, match_func in matching_funcs:
                if not match_func(
                    cfg["recon_req"],
                    cfg.get(key, value),
                    value,
                ):
                    is_match = False
                    break
            if is_match:
                logger.debug("Matched config: %s", cfg)
                yield cfg

    def reconstruct(
        self, output_template: Optional[dict] = None, **matching
    ) -> Iterator[tuple[dict, dict]]:
        output_template = output_template or {}
        cache_key = self._matching_cache_key(output_template, matching)
        configs = self._get_cached_matching(cache_key)

        if configs is None:
            configs = list(
                self._find_matching(
                    self.schema,
                    [{"recon_req": output_template}],
                    **matching,
                )
            )
            self._set_cached_matching(cache_key, configs)

        for cfg in configs:
            yield cfg.pop("recon_req"), cfg
