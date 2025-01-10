import argparse
import sys
from typing import List

import yaml

from conflator import Conflator, ConfigModel

from pproc.config.types import ConfigFactory


def from_outputs(args, overrides: dict):
    with open(args.outputs, "r") as f:
        output_requests = yaml.safe_load(f)

    with open(args.schema, "r") as f:
        schema = yaml.safe_load(f)

    config = ConfigFactory.from_outputs(schema, output_requests, **overrides)
    config_dict = config.model_dump(exclude_none=True, by_alias=True)
    with open(args.out_config, "w") as f:
        yaml.dump(config_dict, f, sort_keys=False)


def from_inputs(args, overrides: dict):
    with open(args.inputs, "r") as f:
        input_requests = yaml.safe_load(f)

    with open(args.schema, "r") as f:
        schema = yaml.safe_load(f)

    config = ConfigFactory.from_inputs(
        schema, args.entrypoint, input_requests, **overrides
    )
    config_dict = config.model_dump(exclude_none=True, by_alias=True)
    with open(args.out_config, "w") as f:
        yaml.dump(config_dict, f, sort_keys=False)


def main(args: List[str] = sys.argv[1:]):
    parser = argparse.ArgumentParser("Generate configuration file for pproc")
    parser.add_argument(
        "--out-config",
        type=str,
        required=True,
        help="Path to output configuration file",
    )

    subparsers = parser.add_subparsers(required=True)
    output_parser = subparsers.add_parser(
        "from_outputs", help="Generate configuration from output requests"
    )
    output_parser.add_argument(
        "--outputs", type=str, required=True, help="Path to output request file"
    )
    output_parser.add_argument(
        "--inputs",
        type=str,
        required=False,
        default=None,
        help="Path to generated input request file",
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
        choices=["pproc-accumulate", "pproc-ensms"],
        help="PProc entrypoint",
    )
    input_parser.add_argument(
        "--inputs", type=str, required=True, help="Path to input request file"
    )
    input_parser.add_argument(
        "--outputs",
        type=str,
        required=False,
        default=None,
        help="Path to generated output request file",
    )
    input_parser.add_argument(
        "--schema",
        type=str,
        required=True,
        help="Path to products schema",
    )
    input_parser.set_defaults(func=from_inputs)
    conflator = Conflator(app_name="pproc-config", model=ConfigModel, argparser=parser)
    overrides = conflator.load()

    args = parser.parse_args(args)
    args.func(args, overrides)


if __name__ == "__main__":
    sys.exit(main())
