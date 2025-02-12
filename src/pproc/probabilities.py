#!/usr/bin/env python3
import sys
from datetime import datetime
import functools
import signal

from meters import ResourceMeter

from pproc import common
from pproc.common.parallel import (
    SynchronousExecutor,
    QueueingExecutor,
    parallel_data_retrieval,
    sigterm_handler,
)
from pproc.common.recovery import Recovery
from pproc.common.param_requester import ParamRequester, ParamConfig
from pproc.prob.parallel import prob_iteration
from pproc.prob.config import BaseProbConfig
from pproc.prob.window_manager import ThresholdWindowManager


class ProbConfig(BaseProbConfig):
    def __init__(self, args, out_keys):
        super().__init__(args, out_keys)
        self.parameters = []
        for pname, popt in self.options["parameters"].items():
            param = ParamConfig(pname, popt, overrides=self.override_input)
            assert param._windows is None, "Use accumulation window configuration"
            self.parameters.append(param)


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    parser = common.default_parser("Compute instantaneous and period probabilites")
    parser.add_argument("-d", "--date", required=True, help="Forecast date")
    parser.add_argument("--in-ens", required=True, help="Source for forecast")
    parser.add_argument(
        "--out-prob", required=True, help="Target for threshold probabilities"
    )
    args = parser.parse_args()
    date = datetime.strptime(args.date, "%Y%m%d%H")

    cfg = ProbConfig(args, ["out_prob"])
    recovery = Recovery(cfg.options["root_dir"], cfg.options, args.recover)
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
            window_manager = ThresholdWindowManager(
                param.window_config(cfg.windows, cfg.steps),
                param.out_keys(cfg.out_keys),
            )
            checkpointed_windows = [
                x["window"] for x in recovery.computed(param=param.name)
            ]
            new_start = window_manager.delete_windows(checkpointed_windows)
            if new_start is None:
                print(f"Recovery: skipping completed param {param.name}")
                continue

            print(f"Recovery: param {param.name} starting from step {new_start}")

            prob_partial = functools.partial(
                prob_iteration, param, recovery, cfg.out_prob
            )
            for keys, retrieved_data in parallel_data_retrieval(
                cfg.n_par_read,
                window_manager.dims,
                [requester],
                cfg.n_par_compute > 1,
                initializer=signal.signal,
                initargs=(signal.SIGTERM, signal.SIG_DFL),
            ):
                step = keys["step"]
                with ResourceMeter(f"Process step {step}"):
                    message_template, data = retrieved_data[0]
                    assert data.ndim == 2

                    completed_windows = window_manager.update_windows(keys, data)
                    for window_id, accum in completed_windows:
                        executor.submit(
                            prob_partial,
                            message_template,
                            window_id,
                            accum,
                            window_manager.thresholds(window_id),
                        )
            executor.wait()

        recovery.clean_file()


if __name__ == "__main__":
    main(sys.argv)
