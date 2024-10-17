import datetime
import functools
import operator
from typing import Dict, List, Optional, Tuple, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


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


def base_request(
    preqs: List[dict],
    accum_dims: List[str],
    pop_keys: List[str] = ["grid", "number", "type", "source"],
) -> Dict[str, Any]:
    base = {}
    for preq in preqs:
        for key, value in preq.items():
            if key in accum_dims + pop_keys:
                continue
            if key in base:
                if base[key] != value:
                    raise ValueError(
                        f"Conflicting values for key {key}: {base[key]} != {value}"
                    )
            else:
                base[key] = value
    return base


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class ParamConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        in_: str = Field(serialization_alias="in")
        dtype: str = "float32"
        in_keys: Optional[dict] = None
        accumulations: dict = {}

    root_dir: Optional[str] = None
    n_par_read: int = 1
    n_par_compute: int = 1
    queue_size: int = 1
    num_members: int = 51
    total_fields: int = 51
    out_keys: Optional[dict] = None
    params: Dict[str, ParamConfig] = {}
    steps: Optional[list] = None
    windows: Optional[list] = None
    sources: dict = {}

    @classmethod
    def from_inputs(
        cls, inputs_path: str, template_path: str, schema_path: Optional[str] = None
    ):
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
            param_config = param_templates.get(param, dict(default_param))
            param_accum = param_config["accumulations"]

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
                param_accum["levelist"] = {"coords": levelists[0]}

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
                in_keys=base_request(preqs, list(param_accum.keys())),
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

        updates = {}
        if global_vars["source"] != "fdb":
            source, loc = global_vars["source"].split(":")
            updates["location"] = loc
        else:
            source = global_vars["source"]

        config.num_members = global_vars["num_members"]
        config.total_fields = total_fields
        config.sources = {
            source: {"ens": [{"type": tp, **updates} for tp in global_vars["type"]]}
        }
        return config

    def from_outputs(outputs, template, schema):
        raise NotImplementedError("Not yet implemented")
