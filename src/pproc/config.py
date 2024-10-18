import argparse
import importlib
import sys
from typing import List

import yaml


def config_from_outputs(args):
    config_generator = importlib.import_module(f"pproc.configs.{args.product}").Config
    config = config_generator.from_outputs(
        args.outputs, args.template, args.schema
    )
    if args.inputs:
        config.outputs(args.inputs)
    config_dict = config.model_dump(exclude_none=True, by_alias=True)
    with open(args.out_config, "w") as f:
        yaml.dump(config_dict, f, sort_keys=False)


def config_from_inputs(args):
    config_generator = importlib.import_module(f"pproc.configs.{args.product}").Config
    config = config_generator.from_inputs(
        args.inputs, args.template, args.schema
    )
    if args.outputs:
        config.outputs(args.outputs)
    config_dict = config.model_dump(exclude_none=True, by_alias=True)
    with open(args.out_config, "w") as f:
        yaml.dump(config_dict, f, sort_keys=False)


def main(args: List[str] = sys.argv[1:]):
    parser = argparse.ArgumentParser("Generate configuration file for pproc")
    parser.add_argument(
        "--product",
        type=str,
        required=True,
        choices=["accum", "ensms"],
        help="Product type",
    )
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
        "--inputs", type=str, required=False, default=None, help="Path to generated input request file"
    )
    output_parser.add_argument(
        "--schema", type=str, required=True, help="Path to products schema"
    )
    output_parser.add_argument(
        "--template",
        type=str,
        required=False,
        help="Path to template configuration file",
    )
    output_parser.set_defaults(func=config_from_outputs)

    input_parser = subparsers.add_parser(
        "from_inputs", help="Generate configuration from input requests"
    )
    input_parser.add_argument(
        "--inputs", type=str, required=True, help="Path to input request file"
    )
    input_parser.add_argument(
        "--outputs", type=str, required=False, default=None, help="Path to generated output request file"
    )
    input_parser.add_argument(
        "--template",
        type=str,
        default=None,
        required=False,
        help="Path to template configuration file",
    )
    input_parser.add_argument(
        "--schema",
        type=str,
        default=None,
        required=False,
        help="Path to products schema",
    )
    input_parser.set_defaults(func=config_from_inputs)

    args = parser.parse_args(args)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
