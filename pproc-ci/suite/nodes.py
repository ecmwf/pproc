import os
import yaml
import pyflow as pf
import wellies as wl
from pwd import getpwuid

SUITE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


def parse_test_directory(test_dir):

    param_file = os.path.join(test_dir, 'parameters.yaml')

    with open(param_file, 'r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    return params


class GenericTestFamily(pf.AnchorFamily):
    def __init__(self, config, test_root, test, **kwargs):
        super().__init__(name=test, **kwargs)

        test_dir = os.path.join(test_root, test)
        test_params = parse_test_directory(test_dir)
        
        for task in test_params['tasks']:
            exec_context = config.execution_contexts[test_params['execution_context']]
            tools = test_params['tools']
            script = [
                config.tools.load(tools),
                pf.FileScript(os.path.join(test_dir, f'{task}.sh')
            ]
            pf.Task(name=task, script=script, submit_arguments=exec_context)


class IntegrationTestsFamily(pf.AnchorFamily):
    def __init__(self, config, **kwargs):
        test_type = 'integration'
        super().__init__(name=test_type, **kwargs)

        with self:
            test_root = os.path.join(SUITE_DIR, 'tests', 'integration')
            for test in os.listdir(test_root):
                GenericTestFamily(config, test_root, test)


class UnitTestsFamily(pf.AnchorFamily):
    def __init__(self, config, **kwargs):
        super().__init__(name='unit', **kwargs)

        with self:
            test_root = os.path.join(SUITE_DIR, 'tests', 'unit')
            for package in os.listdir(test_root):
                script_dir = os.path.join(test_root, package)
                # PackageTestTask(config, package, script_dir)


class MainFamily(pf.AnchorFamily):
    def __init__(self, config, **kwargs):
        super().__init__(name='main', **kwargs)
        with self:

            if 'unit'  in config.test_types:
                UnitTestsFamily(config)

            if 'integration' in config.test_types:
                IntegrationTestsFamily(config)

            if 'regression' in config.test_types:
                RegressionTestsFamily(config)


class InitFamily(pf.AnchorFamily):
    def __init__(self, config, **kwargs):
        super().__init__(name='init', **kwargs)
        with self:
            # install environments and packages
            tools_family = wl.DeployToolsFamily(
                config.environments,
                config.packages,
                config.tools,
                config.installers_dir
            )

            # setup static data (remote/local copy/link)
            deploy_data = wl.DeployDataFamily(config.static_data, config.data_dir)

            fdb = wl.ExecTask(name='bootstrap_fdb', script=pf.FileScript(os.path.join(SUITE_DIR, 'bootstrap_fdb.sh')), exec_context=exec_context['sequential'])
            fdb.triggers = deploy_data.complete & tools_family.complete


class MainSuite(pf.Suite):
    def __init__(self, config):

        variables = {
            'OUTPUT_ROOT': config.output_root,
            'LIB_DIR':     config.lib_dir,
            'DATA_DIR':    config.data_dir,
            'SUITE': config.name,
            'FDB_DIR': config.fdb_dir,
        }
        if config.hostname in ['cca', 'ccb']:
            variables['HOST'] = "%SCHOST:{}%".format(config.hostname)  # in operation, SCHOST will be set by operational server
            variables['ECF_LOGHOST'] = "%HOST%-log"
            variables['ECF_LOGPORT'] = config.logport

        # add your execution limits here
        limits = {
            'work': 20,
        }

        labels = {
            'info': 'PPROC continuous integration suite'
        }

        super().__init__(
            name=config.name,
            host=config.host,
            files=config.deploy_dir,
            include=config.deploy_dir,
            variables=variables,
            limits=limits,
            labels=labels,
            defstatus=pf.state.suspended,
            out=config.log_out,
        )

        limits = {
            k: getattr(self, k) for k in limits.keys()
        }

        with self:

            f_init = InitFamily(config=config, inlimits=self.work)
            f_main = MainFamily(config=config, inlimits=self.work)
            f_main.triggers = f_init.complete