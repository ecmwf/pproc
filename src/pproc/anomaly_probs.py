import sys
from datetime import datetime
import functools
import signal
from typing import Any, Dict

from meters import ResourceMeter

from pproc import common
from pproc.common.parallel import (
    SynchronousExecutor,
    QueueingExecutor,
    parallel_data_retrieval,
    sigterm_handler,
)
from pproc.common.param_requester import ParamRequester, ParamConfig
from pproc.prob.parallel import prob_iteration
from pproc.prob.config import BaseProbConfig
from pproc.prob.window_manager import AnomalyWindowManager
from pproc.prob.climatology import Climatology


class ProbParamConfig(ParamConfig):
    def __init__(self, name, options: Dict[str, Any], overrides: Dict[str, Any] = {}):
        options = options.copy()
        clim_options = options.pop("clim")
        super().__init__(name, options, overrides)
        assert self._windows is None, "Use accumulation window configuration"
        self.clim_param = ParamConfig(f"clim_{name}", clim_options, overrides)


class ProbConfig(BaseProbConfig):
    def __init__(self, args, out_keys):
        super().__init__(args, out_keys)
        self.parameters = [
            ProbParamConfig(pname, popt, overrides=self.override_input)
            for pname, popt in self.options["parameters"].items()
        ]


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    parser = common.default_parser(
        "Compute instantaneous and period probabilites for anomalies"
    )
    parser.add_argument("-d", "--date", required=True, help="Forecast date")
    parser.add_argument("--in-ens", required=True, help="Source for forecast")
    parser.add_argument("--in-clim", required=True, help="Source for climatology")
    parser.add_argument(
        "--out-prob", required=True, help="Target for threshold probabilities"
    )
    args = parser.parse_args()
    date = datetime.strptime(args.date, "%Y%m%d%H")
    cfg = ProbConfig(args, ["out_prob"])

    recovery = common.Recovery(cfg.options["root_dir"], args.config, date, args.recover)
    last_checkpoint = recovery.last_checkpoint()
    executor = (
        SynchronousExecutor()
        if cfg.n_par_compute == 1
        else QueueingExecutor(
            cfg.n_par_compute,
            cfg.window_queue_size,
            initializer=signal.signal,
            initargs=(signal.SIGTERM, signal.SIG_DFL),
        )
    )

    with executor:
        for param in cfg.parameters:
            requester = ParamRequester(
                param,
                cfg.sources,
                args.in_ens,
                cfg.members,
                cfg.total_fields,
            )
            clim = Climatology(
                param.clim_param,
                cfg.sources,
                args.in_clim,
            )
            window_manager = AnomalyWindowManager(
                param.window_config(cfg.windows, cfg.steps),
                param.out_keys(cfg.out_keys),
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

            prob_partial = functools.partial(
                prob_iteration, param, recovery, cfg.out_prob
            )
            for keys, retrieved_data in parallel_data_retrieval(
                cfg.n_par_read,
                window_manager.dims,
                [requester, clim],
                cfg.n_par_compute > 1,
                initializer=signal.signal,
                initargs=(signal.SIGTERM, signal.SIG_DFL),
            ):
                step = keys["step"]
                with ResourceMeter(f"Process step {step}"):
                    message_template, data = retrieved_data[0]
                    assert data.ndim == 2
                    clim_grib_header, clim_data = retrieved_data[1]

                    completed_windows = window_manager.update_windows(
                        keys,
                        data,
                        clim_data[0],
                        clim_data[1],
                    )
                    for window_id, accum in completed_windows:
                        executor.submit(
                            prob_partial,
                            message_template,
                            window_id,
                            accum,
                            window_manager.thresholds(window_id),
                            clim_grib_header,
                        )

            executor.wait()

        recovery.clean_file()


if __name__ == "__main__":
    main(sys.argv)
