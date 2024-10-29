import copy
import functools
import operator
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from pproc.configs.ranges import populate_accums
from pproc.configs.request import Request


def base_request(preqs: List[dict]) -> Request:
    base = {}
    all_keys = set.union(*[set(req.keys()) for req in preqs])
    for key in all_keys:
        values = [req.get(key) for req in preqs]
        if all([x == values[0] for x in values]):
            base[key] = values[0]
    for key in ["source", "grid", "levelist"]:
        if key in all_keys and key not in base:
            raise ValueError(f"Parameter requests contain different values for {key}")
    return base


def check_consistency(
    global_vars: dict, base_req: dict, preqs: List[dict], param_accum: dict
):
    pvars = {"num_members": 0, "type": [], "source": base_req.get("source", "fdb")}

    for preq in preqs:
        pvars["num_members"] += len(preq.get("number", [0]))
        pvars["type"].append(preq["type"])

    # Check total number of fields retrieved is consistent across all parameters
    pvars["total_fields"] = sum(
        [
            functools.reduce(
                operator.mul,
                [
                    len(values)
                    if key not in param_accum.keys() and isinstance(values, list)
                    else 1
                    for key, values in preq.items()
                ],
            )
            for preq in preqs
        ]
    )
    for key, value in pvars.items():
        if key in global_vars and global_vars[key] != value:
            raise ValueError(
                f"Value {value} for {key} is inconsistent with previous requests {global_vars[key]}"
            )
        else:
            global_vars[key] = value


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class ParamConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        in_: str = Field(serialization_alias="in")
        out: Optional[str] = None
        dtype: str = "float32"
        in_keys: Optional[dict] = {}
        out_keys: Optional[dict] = {}
        accumulations: dict = {}

    n_par_read: int = 1
    n_par_compute: int = 1
    queue_size: int = 1
    num_members: int = 51
    total_fields: int = 51
    out_keys: Optional[dict] = {}
    params: Dict[str, ParamConfig] = {}
    steps: Optional[list] = None
    windows: Optional[list] = None
    sources: dict = {}

    @classmethod
    def _from_inputs(
        cls, inputs_path: str, template_path: str, schema_path: Optional[str] = None
    ) -> Self:
        with open(inputs_path, "r") as f:
            input_requests = yaml.safe_load(f)
        param_requests = {}
        for req in input_requests:
            param_requests.setdefault(str(req["param"]), []).append(req)

        with open(template_path, "r") as template_file:
            base_config = yaml.safe_load(template_file)
            param_templates = base_config.pop("params", {})
            default_param = param_templates.pop("default", {"accumulations": {}})
            config = cls(**base_config)

        if schema_path:
            raise NotImplementedError("Schema not yet supported")

        global_vars = {}
        for param, preqs in param_requests.items():
            param_config = copy.deepcopy(default_param)
            param_config.update(param_templates.get(param, {}))
            param_accum = param_config["accumulations"]
            base_req = base_request(preqs)
            populate_accums(param_accum, base_req)

            if len(base_req.get("levelist", [])) > 0:
                param_accum["levelist"] = {
                    "coords": [[x] for x in base_req["levelist"]]
                }

            param_options = cls.ParamConfig(
                in_=param,
                in_keys={
                    k: v
                    for k, v in base_req.items()
                    if k
                    not in list(param_accum.keys())
                    + ["grid", "number", "type", "source"]
                },
                **param_config,
            )
            if base_req.get("grid"):
                param_options.in_keys["interpolate"] = {
                    "grid": base_req["grid"],
                    "intgrid": "none",
                    "legendre-loader": "shmem",
                    "matrix-loader": "file-io",
                }
            config.params[param] = param_options

            check_consistency(global_vars, base_req, preqs, param_accum)

        config.num_members = global_vars["num_members"]
        config.total_fields = global_vars["total_fields"]
        sources = config.sources[global_vars["source"]]
        assert len(sources) == 1, "Only one source is supported"
        source_name = list(sources.keys())[0]
        if len(sources[source_name]) == 0:
            sources[source_name] = [{"type": tp} for tp in global_vars["type"]]
        return config

    @classmethod
    def from_inputs(cls, outputs, template, schema) -> Self:
        return cls._from_inputs(outputs, template, schema)

    def from_outputs(outputs, template, schema):
        raise NotImplementedError("Not yet implemented")

    def outputs(self, path: str):
        raise NotImplementedError("Not yet implemented")

    def inputs(self, path: str):
        raise NotImplementedError("Not yet implemented")
