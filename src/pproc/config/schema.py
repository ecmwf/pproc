import yaml
import copy
import numpy as np
from typing import List, Optional, Generator
from datetime import datetime

from pproc.common.stepseq import fcmonth_to_steprange
from pproc.config.utils import deep_update


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
            else:
                deep_update(config, {key: value})
        return config

    def config_from_output(self, output_request: dict) -> dict:
        output_request = copy.deepcopy(output_request)
        overrides = self.overrides_from_output(output_request)
        config = deep_update(
            self._config_from_output(self.schema, output_request), overrides
        )
        defs = config.pop("defs")
        config.pop("from_inputs", None)
        out = yaml.load(
            yaml.dump(config).format_map({**defs, **output_request}),
            Loader=yaml.SafeLoader,
        )
        return out

    def _config_from_input(
        cls,
        schema: dict,
        input_request: dict,
        configs: Optional[List[dict]] = None,
        **match,
    ):
        if configs is None:
            configs = [{"out": {}, "request": input_request}]

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
                                input_request,
                                [new_fout],
                                **match,
                            )
                        )
                if "*" in schema[key].keys() and len(new_configs) == 0:
                    new_configs = cls._config_from_input(
                        schema[key]["*"], input_request, configs, **match
                    )
                configs = new_configs
            else:
                [deep_update(cfg, {key: value}) for cfg in configs]

        return [cfg for cfg in cls.valid_configs(configs, input_request, **match)]

    def valid_configs(
        cls,
        configs: List[dict],
        input_request: dict,
        **match,
    ) -> Generator:
        for config in configs:
            filled_config = yaml.load(
                yaml.dump(config).format_map(
                    {**config["defs"], **input_request, **config["out"]}
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

            if filled_config["request"] == input_request:
                yield config

    def config_from_input(self, input_request: dict, **match):
        req = copy.deepcopy(input_request)
        overrides = self.overrides_from_input(req)
        for config in self._config_from_input(self.schema, req, **match):
            if config.pop("from_inputs", {}).get("exclude", False):
                continue
            deep_update(config, overrides)
            replace = {
                **config.pop("defs"),
                **req,
                **config.pop("out"),
            }
            filled_config = yaml.load(
                yaml.dump(config).format_map(replace),
                Loader=yaml.SafeLoader,
            )
            yield filled_config

    def overrides_from_output(self, output_request: dict) -> dict:
        overrides = {}
        if steps := output_request.pop("step", None):
            overrides["accumulations"] = {"step": {"coords": []}}
            if isinstance(steps, (int, str)):
                steps = [steps]
            for step in steps:
                try:
                    step = int(step)
                    overrides["accumulations"]["step"]["coords"].append([step])
                except ValueError:
                    steprange = step.split("-")
                    overrides["accumulations"]["step"]["coords"].append(
                        {"from": steprange[0], "to": steprange[1], "by": "{STEP_BY}"}
                    )
        elif fcmonths := output_request.pop("forecastMonth", None):
            overrides["accumulations"] = {"step": {"coords": []}}
            if isinstance(fcmonths, (int, str)):
                fcmonths = [fcmonths]
            for fcmonth in fcmonths:
                start, end = fcmonth_to_steprange(
                    datetime.strptime(output_request["date"], "%Y%m%d"), fcmonth
                ).split("-")
                overrides["accumulations"]["step"]["coords"].append(
                    {"from": start, "to": end, "by": "{STEP_BY}"}
                )
        return overrides

    def overrides_from_input(self, input_request: dict) -> dict:
        overrides = {}
        if members := input_request.pop("number", None):
            overrides["members"] = {
                "start": min(members),
                "end": max(members),
            }
        if steps := input_request.pop("step"):
            try:
                steps = list(map(int, steps))
            except ValueError:
                return overrides
            diff = np.diff(steps)
            assert np.all(diff == diff[0]), "Step intervals must be equal"
            overrides["defs"] = {
                "STEP_START": min(steps),
                "STEP_END": max(steps),
                "STEP_BY": diff[0],
            }
        return overrides
