# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import sys
import os
from typing import List
import logging

import yaml
import json
from pydantic import TypeAdapter

from earthkit.workflows.visualise import visualise

from ppcore.configs import from_outputs as config_from_outputs
from ppcore.configs import from_inputs as config_from_inputs
from ppcore.configs.entrypoint.base import EntrypointConfig
from ppcore.products import product_from_config, graph_from_configs
from ppcore.utils import mars
from ppcore.utils.requests import datacubes, expand
from ppcore.schema.forecast import DatasetDefinition

logging.basicConfig(
    format="%(asctime)s; %(name)s; %(levelname)s - %(message)s",
    level=os.environ.get("PPROC_LOG", "INFO").upper(),
)


def from_outputs(args):
    overrides = {}
    if args.overrides:
        with open(args.overrides, "r") as f:
            overrides = yaml.safe_load(f)

    with open(args.outputs, "r") as f:
        output_requests = yaml.safe_load(f)

    forecast = args.forecast
    if os.path.exists(forecast):
        with open(forecast, "r") as f:
            forecast = TypeAdapter(DatasetDefinition).validate_python(
                **yaml.safe_load(f)
            )

    climatology = args.climatology
    if climatology and os.path.exists(climatology):
        with open(climatology, "r") as f:
            climatology = TypeAdapter(DatasetDefinition).validate_python(
                **yaml.safe_load(f)
            )

    entrypoint_config = EntrypointConfig(
        products=list(
            config_from_outputs(
                output_requests, args.schema, forecast=forecast, climatology=climatology
            )
        ),
        **overrides,
    )
    with open(args.config, "w") as f:
        yaml.dump(
            entrypoint_config.model_dump(exclude_none=True, by_alias=True),
            f,
            sort_keys=False,
        )


def from_inputs(args):
    overrides = {}
    if args.overrides:
        with open(args.overrides, "r") as f:
            overrides = yaml.safe_load(f)

    forecast = args.forecast
    if os.path.exists(forecast):
        with open(forecast, "r") as f:
            forecast = TypeAdapter(DatasetDefinition).validate_python(
                **yaml.safe_load(f)
            )

    climatology = args.climatology
    if climatology and os.path.exists(climatology):
        with open(climatology, "r") as f:
            climatology = TypeAdapter(DatasetDefinition).validate_python(
                **yaml.safe_load(f)
            )

    entrypoint_config = EntrypointConfig(
        products=list(
            config_from_inputs(
                args.restriction,
                args.schema,
                forecast,
                climatology=climatology,
                **overrides,
            )
        ),
        **overrides,
    )
    with open(args.config, "w") as f:
        yaml.dump(
            entrypoint_config.model_dump(exclude_none=True, by_alias=True),
            f,
            sort_keys=False,
        )


def _to_mars(requests: list[dict]) -> str:
    ret = b""
    for req in requests:
        source = req.pop("source", None)
        target = req.pop("target", None)
        # Reverse source/target keys so request can be used to
        # create/extract from the input/output files
        if source and source not in ["fdb", "mars"]:
            req["target"] = f"'{source}'"
        elif target and target != "fdb":
            req["source"] = f"'{target}'"
        ret += mars.to_mars(b"retrieve", req)
        ret += b"\n"
    return ret.decode("utf-8")


def requests(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    entrypoint_config = EntrypointConfig(**config)

    inputs = []
    outputs = []
    for prod_config in entrypoint_config.products:
        product = product_from_config(prod_config)

        inputs.extend(product.in_mars(args.source))
        outputs.extend(product.out_mars(args.target))

    inputs = list(expand(datacubes(inputs), dim="type"))
    outputs = list(expand(datacubes(outputs), dim="type"))

    # TODO: Add method to simplify requests
    if args.inputs:
        _, extension = os.path.splitext(args.inputs)
        with open(args.inputs, "w") as f:
            if args.mars:
                f.write(_to_mars(inputs))
            elif extension == ".json":
                json.dump(inputs, f, sort_keys=False, indent=2)
            else:
                yaml.dump(inputs, f, sort_keys=False)

    if args.outputs:
        _, extension = os.path.splitext(args.outputs)
        with open(args.outputs, "w") as f:
            if args.mars:
                f.write(_to_mars(outputs))
            elif extension == ".json":
                json.dump(outputs, f, sort_keys=False, indent=2)
            else:
                yaml.dump(outputs, f, sort_keys=False)


def plot(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    entrypoint_config = EntrypointConfig(**config)

    graph = graph_from_configs(
        entrypoint_config.products,
        entrypoint_config.input_overrides,
        entrypoint_config.output_overrides,
    )
    visualise(graph, args.file)


def main(args: List[str] = sys.argv[1:]):
    parser = argparse.ArgumentParser("Generate configuration file for PProc")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )

    subparsers = parser.add_subparsers(required=True)
    output_parser = subparsers.add_parser(
        "from_outputs", help="Generate configuration from output requests"
    )
    output_parser.add_argument(
        "--outputs", type=str, required=True, help="Path to output request file"
    )
    output_parser.add_argument(
        "--schema", type=str, required=True, help="Path to products schema"
    )
    output_parser.add_argument(
        "--forecast",
        type=str,
        required=True,
        help="Path to forecast definition, or name of preset definition in schema",
    )
    output_parser.add_argument(
        "--climatology",
        type=str,
        required=False,
        default=None,
        help="Path to climatology definition, or name of preset definition in schema",
    )
    output_parser.add_argument(
        "--overrides",
        type=str,
        required=False,
        help="Path to configuration template for overriding default configuration",
    )
    output_parser.set_defaults(func=from_outputs)

    input_parser = subparsers.add_parser(
        "from_inputs", help="Generate configuration from input requests"
    )
    input_parser.add_argument(
        "--inputs", type=str, required=True, help="Path to input request file"
    )
    input_parser.add_argument(
        "--schema", type=str, required=True, help="Path to products schema"
    )
    input_parser.add_argument(
        "--forecast",
        type=str,
        required=True,
        help="Path to forecast definition, or name of preset definition in schema",
    )
    input_parser.add_argument(
        "--climatology",
        type=str,
        required=False,
        default=None,
        help="Path to climatology definition, or name of preset definition in schema",
    )
    input_parser.add_argument(
        "--restriction",
        type=str,
        required=False,
        help="Restriction to place on output requests e.g. type=em",
    )
    input_parser.add_argument(
        "--overrides",
        type=str,
        required=False,
        help="Path to configuration template for overriding default configuration",
    )
    input_parser.set_defaults(func=from_inputs)

    request_parser = subparsers.add_parser(
        "requests", help="Generate input/output requests from PProc config file"
    )
    request_parser.add_argument(
        "--outputs", type=str, required=False, help="Path to output request file"
    )
    request_parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="Target type to select output requests for",
    )
    request_parser.add_argument(
        "--inputs", type=str, required=False, help="Path to input request file"
    )
    request_parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Source type to select input requests for",
    )
    request_parser.add_argument(
        "--mars",
        action="store_true",
        default=False,
        help="Output in MARS request format",
    )
    request_parser.set_defaults(func=requests)

    plot_parser = subparsers.add_parser(
        "plot", help="Generate plot of earthkit-workflows graph from PProc config file"
    )
    plot_parser.add_argument(
        "--file", type=str, required=False, help="Path to graph html file"
    )
    plot_parser.set_defaults(func=plot)

    parsed = parser.parse_args(args)
    parsed.func(parsed)


if __name__ == "__main__":
    sys.exit(main())
