# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
import sys
import signal

import eccodes
import numpy as np
from meters import ResourceMeter
from conflator import Conflator

from pproc.common.accumulation import Accumulator
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.grib_helpers import construct_message
from pproc.common.io import nan_to_missing
from pproc.common.parallel import (
    create_executor,
    parallel_data_retrieval,
    sigterm_handler,
)
from pproc.common.param_requester import ParamRequester
from pproc.common.recovery import create_recovery, Recovery
from pproc.config.types import AnomalyConfig, ClimParamConfig
from pproc.signi.clim import retrieve_clim


def anomaly_iteration(
    config: AnomalyConfig,
    param: ClimParamConfig,
    recovery: Recovery,
    template: eccodes.GRIBMessage,
    window_id: str,
    accum: Accumulator,
):

    with ResourceMeter(f"{param.name}, window {window_id}: Retrieve climatology"):

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
            "clim",
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
    recovery.add_checkpoint(param=param.name, window=window_id)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-anomaly", model=AnomalyConfig).load()
    cfg.print()
    recovery = create_recovery(cfg)

    with create_executor(cfg.parallelisation) as executor:
        for param in cfg.parameters:
            print(f"Processing {param.name}")
            accum_manager = AccumulationManager.create(
                param.accumulations,
                {
                    **cfg.outputs.default.metadata,
                    **param.metadata,
                },
            )
            checkpointed_windows = [
                x["window"] for x in recovery.computed(param=param.name)
            ]
            accum_manager.delete(checkpointed_windows)

            requester = ParamRequester(param, cfg.sources, cfg.total_fields, "fc")
            anom_partial = functools.partial(anomaly_iteration, cfg, param, recovery)
            for keys, data in parallel_data_retrieval(
                cfg.parallelisation.n_par_read,
                accum_manager.dims,
                [requester],
            ):
                ids = ", ".join(f"{k}={v}" for k, v in keys.items())
                metadata, ens = data[0]
                with ResourceMeter(f"{param.name}, {ids}: Compute accumulation"):
                    completed_windows = accum_manager.feed(keys, ens)
                    del ens
                for window_id, accum in completed_windows:
                    executor.submit(anom_partial, metadata[0], window_id, accum)
            executor.wait()

    recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
