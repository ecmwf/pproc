import os
from os import path
import argparse
import yaml
import pyflow as pf
import wellies as wl
from pwd import getpwuid

import nodes

_description = 'PPOP continuous integration suite'


hpc_install_script = ''' # switch to gnu environment
set +xv
module switch cdt/18.12
set -xv
source /usr/local/etc/ksh_functions/prgenvswitchto
prgenvswitchto gnu
'''


def create_execution_context(hostname, user):

  exec_context = {}
  if hostname in ['cca', 'ccb']:
      host = pf.TrimurtiHost(hostname, user=user)
      host.name = '%SCHOST%'
      host.hostname = '%SCHOST%'
      print('running on the hpc: '+str(host))
      exec_context['sequential'] = wl.HPCSequentialQueue(host, install_script=[hpc_install_script])
      exec_context['fractional'] = wl.HPCFractionalQueue(host, install_script=[hpc_install_script])
      exec_context['parallel'] = wl.HPCParallelQueue(host, install_script=[hpc_install_script])
  elif hostname in ['lxc']:
      host = pf.get_host(hostname)
      print('running on the linux cluster: '+str(host))
      exec_context['sequential'] = wl.LxDefaultQueue(host)
      exec_context['fractional'] = wl.LxParallelQueue(host)
      exec_context['parallel'] = wl.LxParallelQueue(host)
  else:
      host = pf.get_host(hostname)
      print('running on localhost: '+str(host))
      exec_context['sequential'] = wl.LocalHostSequential(host)
      exec_context['fractional'] = wl.LocalHostParallel(host)
      exec_context['parallel'] = wl.LocalHostParallel(host)

  return exec_context


class VirtualEnvTool(wl.Tool):
    def __init__(self, name, lib_dir, depends=[]):
        env_root = path.join(lib_dir, name)
        script = ['source {}'.format(path.join(env_root, 'bin', 'activate')),
                  'export LD_LIBRARY_PATH={}'.format(path.join(env_root, 'lib'))]
        super().__init__(name, depends, script)


def register_tools(virtual_env, packages, modules, lib_dir):

    dependencies = {
        virtual_env: ['python3'],
        'meteokit': [virtual_env],
        'pproc_bundle': ['cmake', 'fftw', virtual_env],
        'mir_python': ['pproc_bundle'],
        'pyfdb': ['pproc_bundle'],
        'eccodeshl': [virtual_env],
        'ppop': [virtual_env],
    }

    tools = []

    tools.append(VirtualEnvTool(virtual_env, lib_dir, depends=dependencies[virtual_env]))

    # env variables
    # env_vars = {
    #     'bin': ('PATH', lib_dir+'/bin'),
    #     'lib': ('LD_LIBRARY_PATH', lib_dir+'/lib'),
    # }
    # for name, values in env_vars.items():
    #     tools.append(wl.EnvTool(name, values[0], values[1], dependencies.get(name, [])))

    # packages
    for package in packages.keys():
        tools.append(wl.Tool(package, dependencies.get(package, [])))

    # modules:
    for module, version in modules.items():
        tools.append(wl.ModuleTool(module, version, dependencies.get(module, [])))

    return wl.ToolStore(tools)


class Config:
    def __init__(self, args):

        with open(args.config_file, 'r') as file:
            options = yaml.load(file, Loader=yaml.FullLoader)

        self.name = options['suite_name']

        self.deploy_dir = options['deploy_dir']
        self.def_file = os.path.join(self.deploy_dir, self.name+'.def')

        # ccb, cca, localhost
        self.hostname = options['host']
        self.user = options['user']
        self.exec_context = create_execution_context(self.hostname, self.user)

        self.output_root = options['output_root']
        self.lib_dir = os.path.join(self.output_root, 'local')
        self.results_dir = os.path.join(self.output_root, 'results')
        self.data_dir = os.path.join(self.output_root, 'data')
        self.fdb_dir = os.path.join(self.output_root, 'fdb')

        self.date = options['date']
        self.windows = [24]
        self.parameters = ['2t']

        self.ppop_env = 'ppop_env'
        self.installer_dir = options['installer_dir']
        self.pip_packages = options.get('pip_packages', [])
        self.packages = options.get('packages', {})
        self.modules = options.get('modules', {})
        self.tools = register_tools(self.ppop_env, self.packages, self.modules, self.lib_dir)

        self.static_data = options.get('static_data', {})


class PPOPCISuite(pf.Suite):
    def __init__(self, config):

        passwd = getpwuid(os.getuid())
        variables = {
                     "HOST": "%SCHOST:{}%".format(config.hostname),  # in operation, SCHOST will be set by operational server
                     'ECF_LOGHOST': "%HOST%-log",
                     "ECF_LOGPORT": 35000+passwd.pw_uid,
                     "PPOP_ENV": config.ppop_env,
                     'OUTPUT_ROOT': config.output_root,
                     'LIB_DIR':     config.lib_dir,
                     'RESULTS_DIR': config.results_dir,
                     'DATA_DIR': config.data_dir,
                     'FDB_DIR': config.fdb_dir,
                    }

        limits = {'work': 20}
        labels = {'info': _description}

        super().__init__(name=config.name,
                         files=config.deploy_dir,
                         include=config.deploy_dir,
                         variables=variables,
                         limits=limits,
                         labels=labels,
                         defstatus=pf.state.suspended)

        self.inlimits += self.work
        with self:
            init_family = nodes.InitFamily(config=config)
            main_family = nodes.MainFamily(config=config)
            main_family.triggers = init_family.complete


def write_definition_file(suite, def_file):
    print('Writing definition file: '+def_file)
    suite_def = suite.ecflow_definition()
    suite_def.check()
    suite_def.save_as_defs(def_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=_description)
    parser.add_argument("config_file", help="yaml configuration file")

    args = parser.parse_args()

    config = Config(args)

    ppop_suite = PPOPCISuite(config)

    ppop_suite.deploy_suite(overwrite=True)

    write_definition_file(ppop_suite, config.def_file)  # and then ecflow_client --host=... --replace=...
