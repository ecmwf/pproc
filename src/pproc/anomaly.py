import functools
import sys
from typing import Union

import eccodes
import numpy as np
from meters import ResourceMeter
from conflator import Conflator

from pproc.common.accumulation import Accumulator
from pproc.common.grib_helpers import construct_message
from pproc.common.io import nan_to_missing, read_template
from pproc.common.parallel import (
    create_executor,
    parallel_data_retrieval,
)
from pproc.common.param_requester import ParamRequester
from pproc.common.recovery import create_recovery, Recovery
from pproc.common.window_manager import WindowManager
from pproc.config.types import AnomalyConfig, AnomalyParamConfig
from pproc.signi.clim import retrieve_clim


def anomaly_iteration(
    config: AnomalyConfig,
    param: AnomalyParamConfig,
    recovery: Recovery,
    template: Union[str, eccodes.GRIBMessage],
    window_id: str,
    accum: Accumulator,
):

    with ResourceMeter(f"{param.name}, window {window_id}: Retrieve climatology"):
        if not isinstance(template, eccodes.GRIBMessage):
            template = read_template(template)

        if "stepRange" in accum.grib_keys():
            steprange = accum.grib_keys()["stepRange"]
        else:
            steprange = template.get("stepRange")

        additional_dims = {"step": steprange}
        if template.get("levtype") == "pl":
            additional_dims["levelist"] = template.get("level")
        clim_accum, _ = retrieve_clim(
            param.clim,
            config.sources,
            ["clim"],
            **additional_dims,
        )
        clim = clim_accum.values
        assert clim is not None

    with ResourceMeter(f"{param.name}, window {window_id}: Compute anomaly"):
        ens = accum.values
        assert ens is not None

        # Anomaly for each ensemble member
        for index, member in enumerate(ens):
            message = construct_message(
                template,
                {
                    **accum.grib_keys(),
                    **config.outputs.ens.metadata,
                    "number": index,
                },
            )
            anom = member - clim[0]
            message.set_array("values", nan_to_missing(message, anom))
            config.outputs.ens.target.write(message)

        # Anomaly for ensemble mean
        ensm_anom = np.mean(ens, axis=0) - clim[0]
        message = construct_message(
            template,
            {
                **accum.grib_keys(),
                **config.outputs.ensm.metadata,
            },
        )
        message.set_array("values", nan_to_missing(message, ensm_anom))
        config.outputs.ensm.target.write(message)

    config.outputs.ens.target.flush()
    config.outputs.ensm.target.flush()
    recovery.add_checkpoint(param.name, window_id)


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)

    cfg = Conflator(app_name="pproc-anomaly", model=AnomalyConfig).load()
    cfg.print()
    recovery = create_recovery(cfg)

    with create_executor(cfg.parallelisation) as executor:
        for param in cfg.parameters:
            print(f"Processing {param.name}")
            window_manager = WindowManager(
                param.accumulations,
                {
                    **cfg.outputs.default.metadata,
                    **param.metadata,
                },
            )
            checkpointed_windows = recovery.computed(param.name)
            new_start = window_manager.delete_windows(checkpointed_windows)
            if new_start is None:
                print(f"Recovery: skipping completed param {param.name}")
                continue

            print(f"Recovery: param {param.name} starting from step {new_start}")

            requester = ParamRequester(
                param, cfg.sources, cfg.members, cfg.total_fields, ["fc"]
            )
            anom_partial = functools.partial(anomaly_iteration, cfg, param, recovery)
            for keys, data in parallel_data_retrieval(
                cfg.parallelisation.n_par_read,
                window_manager.dims,
                [requester],
                cfg.parallelisation.n_par_compute > 1,
            ):
                ids = ", ".join(f"{k}={v}" for k, v in keys.items())
                template, ens = data[0]
                with ResourceMeter(f"{param.name}, {ids}: Compute accumulation"):
                    completed_windows = window_manager.update_windows(keys, ens)
                    del ens
                for window_id, accum in completed_windows:
                    executor.submit(anom_partial, template, window_id, accum)
            executor.wait()

    recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
