import argparse
import functools
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
from meters import ResourceMeter

from pproc.common import parallel
from pproc.common.accumulation import Accumulator
from pproc.common.config import Config, default_parser
from pproc.common.grib_helpers import construct_message
from pproc.common.io import Target, nan_to_missing, target_from_location
from pproc.common.parallel import (
    QueueingExecutor,
    SynchronousExecutor,
    parallel_data_retrieval,
)
from pproc.common.param_requester import ParamConfig, ParamRequester
from pproc.common.recovery import Recovery
from pproc.common.window_manager import WindowManager
from pproc.common.ek_wrappers import GribMetadata
from pproc.signi.clim import retrieve_clim


def get_parser() -> argparse.ArgumentParser:
    description = "Compute weekly anomalies"
    parser = default_parser(description=description)
    parser.add_argument("--in-ens", required=True, help="Input ensemble forecast")
    parser.add_argument("--in-clim", required=True, help="Input climatology")
    parser.add_argument("--out-anom", required=True, help="Output target")
    return parser


class AnomParamConfig(ParamConfig):
    def __init__(self, name, options: Dict[str, Any], overrides: Dict[str, Any] = {}):
        super().__init__(name, options, overrides)
        clim_options = options.copy()
        clim_options.update(options.pop("clim"))
        self.clim_param = ParamConfig(f"clim_{name}", clim_options, overrides)


class AnomConfig(Config):
    def __init__(self, args: argparse.Namespace, verbose: bool = True):
        super().__init__(args, verbose=verbose)

        self.num_members = self.options.get("num_members", 51)
        self.total_fields = self.options.get("total_fields", self.num_members)
        self.out_keys = self.options.get("out_keys", {})
        self.out_ens_keys = {"type": "fcmean", **self.options.get("out_ens_keys", {})}
        self.out_ensm_keys = {"type": "taem", **self.options.get("out_ensm_keys", {})}

        self.params = [
            AnomParamConfig(pname, popt, overrides=self.override_input)
            for pname, popt in self.options["parameters"].items()
        ]

        self.clim_loc: str = args.in_clim
        self.sources = self.options.get("sources", {})

        date = self.options.get("date")
        self.date = None if date is None else datetime.strptime(str(date), "%Y%m%d%H")
        self.root_dir = self.options.get("root_dir", None)

        self.n_par_read = self.options.get("n_par_read", 1)
        self.n_par_compute = self.options.get("n_par_compute", 1)
        self.window_queue_size = self.options.get("queue_size", self.n_par_compute)


def anomaly_iteration(
    config: AnomConfig,
    param: AnomParamConfig,
    target: Target,
    recovery: Optional[Recovery],
    template: GribMetadata,
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
            param.clim_param,
            config.sources,
            config.clim_loc,
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
                    **config.out_ens_keys,
                    "paramId": param.out_paramid,
                    "number": index,
                },
            )
            anom = member - clim[0]
            message.set_array("values", nan_to_missing(message, anom))
            target.write(message)

        # Anomaly for ensemble mean
        ensm_anom = np.mean(ens, axis=0) - clim[0]
        message = construct_message(
            template,
            {
                **accum.grib_keys(),
                **config.out_ensm_keys,
                "paramId": param.out_paramid,
            },
        )
        message.set_array("values", nan_to_missing(message, ensm_anom))
        target.write(message)

    target.flush()
    if recovery is not None:
        recovery.add_checkpoint(param.name, window_id)


def main(args: List[str] = sys.argv[1:]):
    sys.stdout.reconfigure(line_buffering=True)
    parser = get_parser()
    args = parser.parse_args()
    config = AnomConfig(args)
    if config.root_dir is None or config.date is None:
        print("Recovery disabled. Set root_dir and date in config to enable.")
        recovery = None
        last_checkpoint = None
    else:
        recovery = Recovery(config.root_dir, args.config, config.date, args.recover)
        last_checkpoint = recovery.last_checkpoint()
    target = target_from_location(args.out_anom, overrides=config.override_output)
    if config.n_par_compute > 1:
        target.enable_parallel(parallel)
    if recovery is not None and args.recover:
        target.enable_recovery()

    executor = (
        SynchronousExecutor()
        if config.n_par_compute == 1
        else QueueingExecutor(config.n_par_compute, config.window_queue_size)
    )

    with executor:
        for param in config.params:
            window_manager = WindowManager(
                param.window_config([]),
                param.out_keys(config.out_keys),
            )
            if last_checkpoint:
                if param.name not in last_checkpoint:
                    print(f"Recovery: skipping completed param {param.name}")
                    continue
                checkpointed_windows = [
                    recovery.checkpoint_identifiers(x)[1]
                    for x in recovery.checkpoints
                    if param.name in x
                ]
                new_start = window_manager.delete_windows(checkpointed_windows)
                print(f"Recovery: param {param.name} looping from step {new_start}")
                last_checkpoint = None  # All remaining params have not been run

            requester = ParamRequester(
                param,
                config.sources,
                args.in_ens,
                config.num_members,
                config.total_fields,
            )
            anom_partial = functools.partial(
                anomaly_iteration, config, param, target, recovery
            )
            for keys, data in parallel_data_retrieval(
                config.n_par_read,
                window_manager.dims,
                [requester],
            ):
                ids = ", ".join(f"{k}={v}" for k, v in keys.items())
                metadata, ens = data[0]
                with ResourceMeter(f"{param.name}, {ids}: Compute accumulation"):
                    completed_windows = window_manager.update_windows(keys, ens)
                    del ens
                for window_id, accum in completed_windows:
                    executor.submit(anom_partial, metadata[0], window_id, accum)
            executor.wait()

    if recovery is not None:
        recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
