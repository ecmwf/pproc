import sys
from datetime import datetime
import functools
import multiprocessing

from pproc import common
from pproc.common.parallel import (
    SynchronousExecutor,
    QueueingExecutor,
    parallel_data_retrieval,
)
from pproc.prob.parallel import prob_iteration
from pproc.prob.config import ProbConfig
from pproc.prob.window_manager import AnomalyWindowManager
from pproc.prob.climatology import Climatology


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)

    parser = common.default_parser(
        "Compute instantaneous and period probabilites for anomalies"
    )
    parser.add_argument("-d", "--date", required=True, help="Forecast date")
    args = parser.parse_args()
    date = datetime.strptime(args.date, "%Y%m%d%H")
    cfg = ProbConfig(args)

    manager = multiprocessing.Manager()
    recovery = common.Recovery(
        cfg.options["root_dir"], args.config, date, args.recover, manager.Lock()
    )
    last_checkpoint = recovery.last_checkpoint()
    executor = (
        SynchronousExecutor()
        if cfg.n_par_compute == 1
        else QueueingExecutor(cfg.n_par_compute, cfg.window_queue_size)
    )

    with executor:
        for param_name, param_cfg in sorted(cfg.options["parameters"].items()):
            param = common.create_parameter(
                param_name, date, cfg.global_input_cfg, param_cfg, cfg.n_ensembles
            )
            clim = Climatology(
                date, param_cfg["in_paramid"], cfg.global_input_cfg, param_cfg
            )
            window_manager = AnomalyWindowManager(param_cfg, cfg.global_output_cfg)

            if last_checkpoint:
                if param_name not in last_checkpoint:
                    print(f"Recovery: skipping completed param {param_name}")
                    continue
                checkpointed_windows = [
                    recovery.checkpoint_identifiers(x)[1]
                    for x in recovery.checkpoints
                    if param_name in x
                ]
                window_manager.delete_windows(checkpointed_windows)
                print(
                    f"Recovery: param {param_name} looping from step {window_manager.unique_steps[0]}"
                )
                last_checkpoint = None  # All remaining params have not been run

            prob_partial = functools.partial(
                prob_iteration, cfg, param, recovery, False
            )
            for step, retrieved_data in parallel_data_retrieval(
                cfg.n_par_read,
                window_manager.unique_steps,
                [param, clim],
                cfg.n_par_compute > 1,
            ):
                with common.ResourceMeter(f"Process step {step}"):
                    message_template, data = retrieved_data[0]
                    clim_grib_header, clim_data = retrieved_data[1]

                    completed_windows = window_manager.update_windows(
                        step,
                        data,
                        clim_data[clim.get_type_index("em")],
                        clim_data[clim.get_type_index("es")],
                    )
                    for window_id, window in completed_windows:
                        executor.submit(
                            prob_partial,
                            message_template,
                            window_id,
                            window,
                            window_manager.thresholds(window_id),
                            additional_headers=clim_grib_header,
                        )

            executor.wait()

        recovery.clean_file()


if __name__ == "__main__":
    main(sys.argv)
