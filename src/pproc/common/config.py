import argparse
import yaml


def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    return dict(map(lambda s: s.split('='), items))


def default_parser(description):
    """
    Create a default parser with two options: a yaml config file and a set option to update the entries of the yaml config
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config', required=True, help='YAML configuration file')
    parser.add_argument('-s', '--set',
                        metavar="KEY=VALUE",
                        nargs='+',
                        help="Set a number of key-value pairs "
                             "(do not put spaces before or after the = sign). "
                             "If a value contains spaces, you should define "
                             "it with double quotes: "
                             'foo="this is a sentence". Note that '
                             "values are always treated as strings.")
    parser.add_argument('--recover', action="store_true", default=True,
                        help='Continue from last checkpoint in recovery file.')
    parser.add_argument('--no-recover', action="store_false", dest="recover",
                        help='Ignore checkpoints and recompute from beginning.')

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


class Config():
    def __init__(self, args, verbose=True):

        with open(args.config, 'r') as f:
            self.options = yaml.safe_load(f)
        
        if args.set:
            values_to_set = parse_vars(args.set)
            for key, value in values_to_set.items():
                nested_set(self.options, key.split('.'), value)

        if verbose:
            print(yaml.dump(self.options))
