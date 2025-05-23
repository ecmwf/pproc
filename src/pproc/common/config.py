# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import yaml


def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    return dict(map(lambda s: s.split("="), items))


def parse_var_strs(items):
    """
    Parse a list of comma-separated lists of key-value pairs and return a dictionary
    """
    return parse_vars(sum((s.split(",") for s in items if s), start=[]))


def default_parser(description):
    """
    Create a default parser with two options: a yaml config file and a set option to update the entries of the yaml config
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-c", "--config", required=True, help="YAML configuration file")
    parser.add_argument(
        "-s",
        "--set",
        metavar="KEY=VALUE",
        nargs="+",
        help="Set a number of key-value pairs "
        "(do not put spaces before or after the = sign). "
        "If a value contains spaces, you should define "
        "it with double quotes: "
        'foo="this is a sentence". Note that '
        "values are always treated as strings.",
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        default=True,
        help="Continue from last checkpoint in recovery file.",
    )
    parser.add_argument(
        "--no-recover",
        action="store_false",
        dest="recover",
        help="Ignore checkpoints and recompute from beginning.",
    )
    parser.add_argument(
        "--override-input",
        action="append",
        default=[],
        metavar="KEY=VALUE,...",
        help="Override input requests with these keys",
    )
    parser.add_argument(
        "--override-output",
        action="append",
        default=[],
        metavar="KEY=VALUE,...",
        help="Override output metadata with these keys",
    )

    return parser


def nested_set(dic, keys, value):
    """
    Set the values from a nested dictionnary using a list of keys
    """
    for key in keys[:-1]:
        dic = dic[key]
    val_in_dic = dic.get(keys[-1], None)
    val_type = type(val_in_dic) if val_in_dic else None
    dic[keys[-1]] = val_type(value)


class Config:
    def __init__(self, args, verbose=True):
        with open(args.config, "r") as f:
            self.options = yaml.safe_load(f)

        if args.set:
            values_to_set = parse_vars(args.set)
            for key, value in values_to_set.items():
                nested_set(self.options, key.split("."), value)

        self.override_input = parse_var_strs(args.override_input)
        self.override_output = parse_var_strs(args.override_output)

        if verbose:
            print(yaml.dump(self.options, sort_keys=False))
