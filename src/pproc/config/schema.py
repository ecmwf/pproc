import yaml
import copy
import numpy as np
from typing import List, Optional, Generator, Iterator
from datetime import datetime

from pproc.common.stepseq import fcmonth_to_steprange
from pproc.config.utils import deep_update, update_request


class Schema:
    def __init__(self, schema_path: str):
        with open(schema_path, "r") as f:
            schema = yaml.safe_load(f)
        self.schema = self.expand(schema)

    def expand(cls, schema: dict) -> dict:
        expanded = {}
        for keys, value in schema.items():
            if cls.is_subschema(keys):
                expanded.setdefault(keys, {})
                for sub_keys, sub_values in value.items():
                    sub_expanded = cls.expand(sub_values)
                    for sub_key in sub_keys.split("/"):
                        expanded[keys][sub_key] = sub_expanded
            else:
                expanded[keys] = value
        return expanded

    def is_subschema(cls, key: str) -> bool:
        return "filter" in key

    def subschema(cls, key: str, schema: dict, request: dict) -> dict:
        _, mars_key = key.split(":")
        filter_value = request[mars_key]
        ret = schema.get(filter_value, schema.get("*", None))
        if ret is None:
            raise ValueError(
                f"Filter value {filter_value} not found in schema, and no default provided"
            )
        return ret

    @classmethod
    def validate_request(cls, request: dict) -> dict:
        out = copy.deepcopy(request)
        if isinstance(out["param"], int):
            out["param"] = str(out["param"])
        elif np.ndim(out["param"]) > 0:
            out["param"] = [str(param) for param in out["param"]]
        if isinstance(out["type"], list) and len(out["type"]) > 1:
            raise ValueError("Multiple types in request are not allowed")
        return out

    def _config_from_output(
        cls, sub_schema: dict, output_request: dict, config: Optional[dict] = None
    ) -> dict:
        config = (
            {"request": copy.deepcopy(output_request)} if config is None else config
        )
        for key, value in sub_schema.items():
            if cls.is_subschema(key):
                cls._config_from_output(
                    cls.subschema(key, value, output_request), output_request, config
                )
            elif key == "request":
                config["request"] = update_request(config.get("request", {}), value)
            else:
                deep_update(config, {key: value})

        return config

    def config_from_output(self, output_request: dict) -> dict:
        output_request = self.validate_request(output_request)
        config = self._config_from_output(self.schema, output_request)
        reqs = (
            config["request"]
            if isinstance(config["request"], list)
            else [config["request"]]
        )
        out = yaml.load(
            yaml.dump(config).format_map({**output_request}),
            Loader=yaml.SafeLoader,
        )
        return out

    def combined_params(self) -> Iterator:
        params = self.schema.get("filter:param", {})
        for paramId, config in params.items():
            req_params = config.get("request", {}).get("param", [])
            if isinstance(req_params, list) and len(req_params) > 1:
                yield int(paramId), list(map(int, req_params))

    def valid_configs(
        cls,
        configs: List[dict],
        input_requests: list[dict],
        **match,
    ) -> Generator:
        for config in configs:
            filled_config = yaml.load(
                yaml.dump(config).format_map(
                    {**config["defs"], **input_requests[0], **config["out"]}
                ),
                Loader=yaml.SafeLoader,
            )

            is_match = True
            for key, value in match.items():
                if filled_config.get(key, value) != value:
                    is_match = False
                    break
            if not is_match:
                continue

            if filled_config["request"] == input_requests:
                yield config

    def _config_from_input(
        cls,
        schema: dict,
        input_requests: list[dict],
        configs: Optional[List[dict]] = None,
        **match,
    ):
        if configs is None:
            configs = [{"out": {}, "request": input_requests}]

        for key, value in schema.items():
            if cls.is_subschema(key):
                filter_key = key.split(":")[1]
                new_configs = []
                for filter_value in value.keys():
                    if filter_value == "*":
                        continue
                    for fout in configs:
                        new_fout = copy.deepcopy(fout)
                        if (
                            new_fout["out"].setdefault(filter_key, filter_value)
                            != filter_value
                        ):
                            continue
                        new_configs.extend(
                            cls._config_from_input(
                                schema[key][filter_value],
                                input_requests,
                                [new_fout],
                                **match,
                            )
                        )
                if "*" in schema[key].keys() and len(new_configs) == 0:
                    new_configs = cls._config_from_input(
                        schema[key]["*"], input_requests, configs, **match
                    )
                configs = new_configs
            elif key == "request":
                [
                    deep_update(cfg, {key: update_request(cfg.get("request"), value)})
                    for cfg in configs
                ]
            else:
                [deep_update(cfg, {key: value}) for cfg in configs]

        return [cfg for cfg in cls.valid_configs(configs, input_requests, **match)]

    def config_from_input(self, input_requests: list[dict], **match):
        reqs = [self.validate_request(x) for x in input_requests]
        reqs.sort(key=lambda x: x["type"])
        overrides = self.overrides_from_input(reqs)
        for config in self._config_from_input(self.schema, reqs, **match):
            if config.pop("from_inputs", {}).get("exclude", False):
                continue
            deep_update(config, overrides)
            defs = config.pop("defs")
            replace = {
                **defs,
                **reqs[0],
                **config.pop("out"),
            }
            filled_config = yaml.load(
                yaml.dump(config).format_map(replace),
                Loader=yaml.SafeLoader,
            )
            reqs = (
                filled_config["request"]
                if isinstance(config["request"], list)
                else [filled_config["request"]]
            )
            for req in reqs:
                if grid := req.pop("interp_grid", None):
                    req["interpolate"] = {
                        "grid": grid,
                        **defs["interp_keys"],
                    }
            yield filled_config

    def overrides_from_input(self, reqs: list[dict]) -> dict:
        overrides = {}
        for req in reqs:
            if members := req.pop("number", None):
                if isinstance(members, (int, str)):
                    members = [members]
                members = list(map(int, members))
                set_members = {
                    "start": min(members),
                    "end": max(members),
                }
                if overrides.setdefault("members", set_members) != set_members:
                    raise ValueError(f"Multiple member ranges in {reqs}")
                # Ensure preset in schema is not used
                overrides.setdefault("total_fields", 0)
            if steps := req.pop("step"):
                try:
                    steps = list(map(int, steps))
                except ValueError:
                    return overrides
                steps.sort()
                diff = np.diff(steps)
                assert np.all(diff == diff[0]), "Step intervals must be equal"
                set_steps = {
                    "STEP_START": min(steps),
                    "STEP_END": max(steps),
                    "STEP_BY": diff[0],
                }
                if overrides.setdefault("defs", set_steps) != set_steps:
                    raise ValueError(f"Multiple step ranges in {reqs}")
        return overrides
