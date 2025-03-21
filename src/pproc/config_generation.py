import argparse
import sys
import os
from typing import List

import yaml
import json


from pproc.config.factory import ConfigFactory
from pproc.schema.schema import Schema
from pproc.common import mars


def from_outputs(args):
    overrides = {}
    if args.overrides:
        with open(args.overrides, "r") as f:
            overrides = yaml.safe_load(f)

    with open(args.outputs, "r") as f:
        output_requests = yaml.safe_load(f)

    schema = Schema.from_file(args.schema)
    config = ConfigFactory.from_outputs(schema, output_requests, **overrides)
    config_dict = config.model_dump(exclude_none=True, by_alias=True)
    with open(args.config, "w") as f:
        yaml.dump(config_dict, f, sort_keys=False)


def from_inputs(args):
    overrides = {}
    if args.overrides:
        with open(args.overrides, "r") as f:
            overrides = yaml.safe_load(f)

    with open(args.inputs, "r") as f:
        input_requests = yaml.safe_load(f)

    schema = Schema.from_file(args.schema)
    config = ConfigFactory.from_inputs(
        schema, args.entrypoint, input_requests, **overrides
    )
    config_dict = config.model_dump(exclude_none=True, by_alias=True)
    with open(args.config, "w") as f:
        yaml.dump(config_dict, f, sort_keys=False)


def _to_mars(requests: list[dict]) -> str:
    ret = b""
    for req in requests:
        req.pop("source", None)
        req.pop("target", None)
        ret += mars.to_mars(b"retrieve", req)
        ret += b"\n"
    return ret.decode("utf-8")


def requests(args):
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = ConfigFactory.from_dict(args.entrypoint, **config_dict)

    if args.inputs:
        _, extension = os.path.splitext(args.inputs)
        inputs = list(config.in_mars(args.source))
        with open(args.inputs, "w") as f:
            if args.mars:
                f.write(_to_mars(inputs))
            elif extension == ".json":
                json.dump(inputs, f, sort_keys=False, indent=2)
            else:
                yaml.dump(inputs, f, sort_keys=False)

    if args.outputs:
        _, extension = os.path.splitext(args.outputs)
        outputs = list(config.out_mars(args.target))
        with open(args.outputs, "w") as f:
            if args.mars:
                f.write(_to_mars(outputs))
            elif extension == ".json":
                json.dump(outputs, f, sort_keys=False, indent=2)
            else:
                yaml.dump(outputs, f, sort_keys=False)


def main(args: List[str] = sys.argv[1:]):
    parser = argparse.ArgumentParser("Generate configuration file for pproc")
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
        "--overrides",
        type=str,
        required=False,
        help="Path to configuration template for overriding default configuration",
    )
    output_parser.add_argument(
        "--schema", type=str, required=True, help="Path to products schema"
    )
    output_parser.set_defaults(func=from_outputs)

    input_parser = subparsers.add_parser(
        "from_inputs", help="Generate configuration from input requests"
    )
    input_parser.add_argument(
        "--entrypoint",
        type=str,
        required=True,
        choices=["pproc-accumulate", "pproc-ensms", "pproc-monthly-stats"],
        help="PProc entrypoint",
    )
    input_parser.add_argument(
        "--inputs", type=str, required=True, help="Path to input request file"
    )
    input_parser.add_argument(
        "--schema",
        type=str,
        required=True,
        help="Path to products schema",
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
        "--entrypoint", type=str, required=True, help="PProc entrypoint"
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

    args = parser.parse_args(args)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
