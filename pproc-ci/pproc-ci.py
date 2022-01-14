import os
from argparse import ArgumentParser
import yaml

import wellies as wl

from suite.nodes import MainSuite

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


class Config:
    def __init__(self, args):

        with open(args.config_file, 'r') as file:
            options = yaml.load(file, Loader=yaml.FullLoader)

        self.name = options['suite_name']
        self.log_out = options['log_out']

        self.deploy_dir = options['deploy_dir']
        self.def_file = os.path.join(self.deploy_dir, 'pproc-ci.def')

        self.user = options['user']

        self.hostname = options['hostname']
        self.host = wl.get_host(self.hostname, self.user)
        self.logport = options['logport']

        self.execution_contexts = parse_host_config(args.hosts_config)

        self.output_root = options['output_root']
        self.lib_dir = options['lib_dir']
        self.data_dir = options['data_dir']
        self.fdb_dir = os.path.join(self.output_root, 'fdb')

        self.log_out = options['log_out']

        self.modules = parse_tools_file(args.tools_config, 'modules')
        self.packages = parse_tools_file(args.tools_config, 'packages')
        self.env_vars = parse_tools_file(args.tools_config, 'env_variables')
        self.environments = parse_tools_file(args.tools_config, 'environments')
        self.static_data = parse_tools_file(args.tools_config, 'static_data')

        self.tools = wl.register_tools(
            self.lib_dir,
            self.modules,
            self.packages,
            self.environments,
            self.env_vars,
        )

        self.date = options['date']
        self.windows = [24]
        self.parameters = ['2t']

        self.test_types = options['test_types']

def parse_host_config(filepath):
    """parse hosts definition file and create execution context manager.
    """

    if filepath is None:
        return {}

    with open(filepath, 'r') as file:
        hosts_params = yaml.load(file, Loader=yaml.FullLoader)
    try:
        execution_contexts = hosts_params['execution_contexts']
    except KeyError:
        raise KeyError(
            f'Host config file must contain an "execution_contexts" '
            f'entry at top level. Please make sure there is one in {filepath}'
        )

    return execution_contexts


def parse_tools_file(filepath, configKey):

    if filepath is None:
        return {}

    with open(filepath, 'r') as f1:
        tools_params = yaml.load(f1, Loader=yaml.FullLoader)

    return tools_params.get(configKey, {})


def get_parser() -> ArgumentParser:
    description = (
        "\n"
        "Generate required files for a pyflow suite project."
        "\n"
    )
    parser = ArgumentParser(
        usage='%(prog)s <CONFIG_FILE>',
        description=description,
    )
    parser.add_argument(
        'config_file', metavar='CONFIG_FILE',
        help='configuration file path'
    )
    parser.add_argument(
        '--hosts', metavar='HOSTS_FILE', dest='hosts_config', default='hosts.yaml',
        help='Hosts and execution context configuration file path',
    )
    parser.add_argument(
        '--tools', metavar='TOOLS_FILE', dest='tools_config', default='tools_data.yaml',
        help='Tools and static data configuration file path',
    )
    return parser


if __name__ == '__main__':

    # create parser and parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # create config object and suite
    config = Config(args)
    suite = MainSuite(config)
    print(suite)

    # deploy suite scripts and definition file
    suite.deploy_suite(overwrite=True)
    wl.write_definition_file(suite, config.def_file)
