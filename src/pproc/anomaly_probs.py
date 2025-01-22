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
from pproc.common.parameter import create_parameter
from pproc.prob.parallel import prob_iteration
from pproc.prob.config import ProbConfig
from pproc.prob.window_manager import AnomalyWindowManager
from pproc.prob.climatology import Climatology


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    parser = common.default_parser(
        "Compute instantaneous and period probabilites for anomalies"
    )
    parser.add_argument("-d", "--date", required=True, help="Forecast date")
    parser.add_argument(
        "--out_prob", required=True, help="Target for threshold probabilities"
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
        for param_name, param_cfg in sorted(cfg.options["parameters"].items()):
            param = create_parameter(
                param_name,
                date,
                cfg.global_input_cfg,
                param_cfg,
                cfg.n_ensembles,
                cfg.override_input,
            )
            clim = Climatology(
                date,
                param_cfg["in_paramid"],
                cfg.global_input_cfg,
                param_cfg,
                cfg.override_input,
            )
            window_manager = AnomalyWindowManager(param_cfg, cfg.global_output_cfg)
            checkpointed_windows = recovery.computed(param_name)
            new_start = window_manager.delete_windows(checkpointed_windows)
            if new_start is None:
                print(f"Recovery: skipping completed param {param_name}")
                continue

            print(f"Recovery: param {param_name} starting from step {new_start}")

            prob_partial = functools.partial(
                prob_iteration, param, recovery, cfg.out_prob
            )
            for keys, retrieved_data in parallel_data_retrieval(
                cfg.n_par_read,
                window_manager.dims,
                [param, clim],
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
                        clim_data.sel(type="em"),
                        clim_data.sel(type="es"),
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
