import os
import pyflow as pf
import wellies as wl


SUITE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))



class ParameterFamily(pf.AnchorFamily):
    def __init__(self, config, parameter, **kwargs):
        super().__init__(name=parameter, **kwargs)

        env_script = config.tools.script([config.ppop_env])

        fdb_scripts = [env_script, pf.FileScript(os.path.join(SUITE_DIR, 'populate_fdb.sh'))]
        compute_scripts = [env_script, pf.FileScript(os.path.join(SUITE_DIR, 'efi_sot.sh'))]

        with self:
            (
                wl.ExecTask(name='populate_fdb', exec_context=config.exec_context['sequential'], script=fdb_scripts)
                >>
                wl.ExecTask(name='efi_sot', exec_context=config.exec_context['sequential'], script=compute_scripts)
            )


class MainFamily(pf.AnchorFamily):
    def __init__(self, config, **kwargs):
        super().__init__(name='main', **kwargs)

        with self:
            with pf.Family(name='test_efi_sot'):
                for param in config.parameters:
                    ParameterFamily(config, param)
                            


class VirtualEnvTask(wl.ExecTask):
    def __init__(self, name, tools, packages, **kwargs):
        scripts = [
            'export https_proxy=http://proxy.ecmwf.int:3333',
            tools.dependencies_script(name),
            'rm -rf $LIB_DIR/{}'.format(name),
            'python3 -m venv $LIB_DIR/{} --system-site-packages'.format(name),
            tools.tools[name].script,
        ]
        for package in packages:
            scripts.append('TMPDIR=$SCRATCH/tmp python3 -m pip install {}'.format(package))
        super().__init__(name=name, script=scripts, **kwargs)


class InitFamily(pf.AnchorFamily):
    def __init__(self, config, **kwargs):
        super().__init__(name='init', **kwargs)
        exec_context = config.exec_context
        with self:
            
            env_task = VirtualEnvTask(config.ppop_env, config.tools, config.pip_packages, exec_context=exec_context['sequential'])
            tools_family = wl.DeployToolsFamily(config.packages, config.tools, config.installer_dir, exec_context['fractional'])
            tools_family.triggers = env_task.complete
            
            data_family = wl.DeployDataFamily(config.static_data, config.data_dir, exec_context['sequential'])

            fdb = wl.ExecTask(name='bootstrap_fdb', script=pf.FileScript(os.path.join(SUITE_DIR, 'bootstrap_fdb.sh')), exec_context=exec_context['sequential'])
            fdb.triggers = data_family.complete & tools_family.complete
