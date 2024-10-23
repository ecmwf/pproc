import copy
import functools
import operator
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from pproc.configs.ranges import populate_accums
from pproc.configs.request import Request


def parse_requests(inputs: List[dict]) -> Tuple[Dict[str, Any], Dict[str, List[dict]]]:
    param_reqs = {}
    for req in inputs:
        param_reqs.setdefault(str(req["param"]), []).append(req)

    global_vars = {}
    for param, preqs in param_reqs.items():
        # Check for consistency between different parameters for common fields
        pmembers = sum([len(preq.get("number", [0])) for preq in preqs])
        if pmembers != global_vars.setdefault("num_members", pmembers):
            raise ValueError(
                f"Number of members do not match: {global_vars['num_members']} != {pmembers}"
            )

        ptypes = [preq["type"] for preq in preqs]
        if ptypes != global_vars.setdefault("type", ptypes):
            raise ValueError(
                f"Parameter types do not match: {global_vars['type']} != {ptypes}"
            )

        psources = list(set(preq.get("source", "fdb") for preq in preqs))
        if len(psources) > 1:
            raise ValueError(
                f"Different sources found in requests for param {param}: {psources}"
            )
        if psources[0] != global_vars.setdefault("source", psources[0]):
            raise ValueError(
                f"Sources do not match: {global_vars['source']} != {psources[0]}"
            )

    return global_vars, param_reqs


def base_request(preqs: List[dict]) -> Request:
    base = {}
    all_keys = set.union(*[set(req.keys()) for req in preqs])
    for key in all_keys:
        values = [req.get(key) for req in preqs]
        if all([x == values[0] for x in values]):
            base[key] = values[0]
    return base


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
        global_vars, param_requests = parse_requests(input_requests)

        with open(template_path, "r") as template_file:
            base_config = yaml.safe_load(template_file)
            param_templates = base_config.pop("params", {})
            default_param = param_templates.pop("default", {"accumulations": {}})
            config = cls(**base_config)

        if schema_path:
            raise NotImplementedError("Schema not yet supported")

        total_fields = None
        for param, preqs in param_requests.items():
            param_config = copy.deepcopy(default_param)
            param_config.update(param_templates.get(param, {}))
            param_accum = param_config["accumulations"]
            base_req = base_request(preqs)
            populate_accums(param_accum, base_req)

            # Check grid and levelist are consistent with requests for same parameter
            grid = list(set(preq.pop("grid", None) for preq in preqs))
            if len(grid) > 1:
                raise ValueError(
                    f"Different grids found in requests for param {param}: {grid}"
                )
            levelists = [preq.get("levelist", []) for preq in preqs]
            if not all([levelists[0] == x for x in levelists]):
                raise ValueError(
                    f"Different levelist found in requests for param {param}: {levelists}"
                )
            elif len(levelists[0]) > 0:
                param_accum["levelist"] = {"coords": [[x] for x in levelists[0]]}

            # Check total number of fields retrieved is consistent across all parameters
            pfields = sum(
                [
                    functools.reduce(
                        operator.mul,
                        [
                            len(values)
                            if key not in param_accum.keys()
                            and isinstance(values, list)
                            else 1
                            for key, values in preq.items()
                        ],
                    )
                    for preq in preqs
                ]
            )
            if not total_fields:
                total_fields = pfields
            elif pfields != total_fields:
                raise ValueError(
                    f"Number of fields do not match: {total_fields} != {pfields}"
                )

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
            if grid[0]:
                param_options.in_keys["interpolate"] = {
                    "grid": grid[0],
                    "intgrid": "none",
                    "legendre-loader": "shmem",
                    "matrix-loader": "file-io",
                }
            config.params[param] = param_options

        config.num_members = global_vars["num_members"]
        config.total_fields = total_fields
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
