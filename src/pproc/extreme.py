#!/usr/bin/env python3
#
# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.


import numpy as np
import sys
from datetime import datetime
import functools
import signal

import eccodes
from meteokit import extreme
from pproc import common
from pproc.common.parallel import (
    SynchronousExecutor,
    QueueingExecutor,
    parallel_data_retrieval,
    sigterm_handler, 
    shared_list, 
    shared_lock
)


class ExtremeVariables:
    def __init__(self, efi_cfg):
        self.eps = float(efi_cfg["eps"])
        self.sot = list(map(int, efi_cfg["sot"]))


def read_clim(fdb, climatology, window, n_clim=101):

    req = climatology["clim_keys"].copy()
    assert "date" in req
    req["time"] = "0000"
    req["quantile"] = ["{}:100".format(i) for i in range(n_clim)]
    if window.name in climatology.get("steps", {}):
        req["step"] = climatology["steps"][window.name]
    else:
        req["step"] = window.name

    print("Climatology request: ", req)
    da_clim = common.fdb_read(fdb, req)
    assert da_clim.values.shape[0] == n_clim
    da_clim_sorted = da_clim.reindex(quantile=[f"{x}:100" for x in range(n_clim)])
    print(da_clim_sorted)

    return np.asarray(da_clim_sorted.values), da_clim.attrs["grib_template"]


def extreme_template(window, template_fc, template_clim):

    template_ext = template_fc.copy()

    for key, value in window.config_grib_header.items():
        template_ext[key] = value

    # EFI specific stuff
    template_ext["stepRange"] = window.name
    if int(template_ext["timeRangeIndicator"]) == 3:
        if template_ext["numberIncludedInAverage"] == 0:
            template_ext["numberIncludedInAverage"] = len(window.steps)
        template_ext["numberMissingFromAveragesOrAccumulations"] = 0

    # set clim keys
    clim_keys = [
        "powerOfTenUsedToScaleClimateWeight",
        "weightAppliedToClimateMonth1",
        "firstMonthUsedToBuildClimateMonth1",
        "lastMonthUsedToBuildClimateMonth1",
        "firstMonthUsedToBuildClimateMonth2",
        "lastMonthUsedToBuildClimateMonth2",
        "numberOfBitsContainingEachPackedValue",
    ]
    for key in clim_keys:
        template_ext[key] = template_clim[key]

    # set fc keys
    fc_keys = [
        "date",
        "subCentre",
        "totalNumber",
    ]
    for key in fc_keys:
        template_ext[key] = template_fc[key]

    return template_ext


def efi_template(template):
    template_efi = template.copy()
    template_efi["marsType"] = 27
    template_efi["efiOrder"] = 0
    template_efi["number"] = 0
    return template_efi


def efi_template_control(template):
    template_efi = template.copy()
    template_efi["marsType"] = 28
    template_efi["efiOrder"] = 0
    template_efi["totalNumber"] = 1
    template_efi["number"] = 0
    return template_efi


def sot_template(template, sot):
    template_sot = template.copy()
    template_sot["marsType"] = 38
    template_sot["number"] = sot
    if sot == 90:
        template_sot["efiOrder"] = 99
    elif sot == 10:
        template_sot["efiOrder"] = 1
    else:
        raise Exception(
            "SOT value '{sot}' not supported in template! Only accepting 10 and 90"
        )
    return template_sot


def efi_sot(cfg, param, climatology, efi_vars, recovery, template_filename, window_id, window
):
    with common.ResourceMeter(f"Window {window.suffix}, computing EFI/SOT"):

        message_template = (
            template_filename
            if isinstance(template_filename, eccodes.highlevel.message.GRIBMessage)
            else common.io.read_template(template_filename)
        )

        clim, template_clim = read_clim(common.io.fdb(), climatology, window)
        print(f"Climatology array: {clim.shape}")

        template_extreme = extreme_template(window, message_template, template_clim)

        control_index = param.get_type_index("cf", default=None)
        if control_index is not None:
            efi_control = extreme.efi(clim, window.step_values[control_index], efi_vars.eps)
            template_efi = efi_template_control(template_extreme)
            common.write_grib(cfg.out_efi, template_efi, efi_control)

        efi = extreme.efi(clim, window.step_values, efi_vars.eps)
        template_efi = efi_template(template_extreme)
        common.write_grib(cfg.out_efi, template_efi, efi)

        sot = {}
        for perc in efi_vars.sot:
            sot[perc] = extreme.sot(clim, window.step_values, perc, efi_vars.eps)
            template_sot = sot_template(template_extreme, perc)
            common.write_grib(cfg.sot_target, template_sot, sot[perc])

        common.io.fdb.flush()
        recovery.add_checkpoint(param.name, window_id)


class ConfigExtreme(common.Config):
    def __init__(self, args):
        super().__init__(args)

        self.fc_date = datetime.strptime(str(self.options["fc_date"]), "%Y%m%d%H")

        if isinstance(self.options["members"], dict):
            self.members = range(self.options["members"]["start"], self.options["members"]["end"] + 1)
        else:
            self.members = int(self.options["members"])
        self.n_par_compute = self.options.get("n_par_compute", 1)
        self.n_par_read = self.options.get("n_par_read", 1)
        self.window_queue_size = self.options.get("queue_size", self.n_par_compute)

        self.root_dir = self.options["root_dir"]

        self.global_input_cfg = self.options.get("global_input_keys", {})
        self.global_output_cfg = self.options.get("global_output_keys", {})

        for attr in ["out_efi", "out_sot"]:
            location = getattr(args, attr)
            target = common.io.target_from_location(location)
            if self.n_par_compute > 1 and type(target) in [common.io.FileTarget, common.io.FileSetTarget]:
                target.track_truncated = shared_list()
            self.__setattr__(attr, target)

        print(f"Forecast date is {self.fc_date}")
        print(f"Root directory is {self.root_dir}")


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    parser = common.default_parser(
        "Compute EFI and SOT from forecast and climatology"
    )
    parser.add_argument(
        "--out_efi", required=True, help="Target for EFI"
    )
    parser.add_argument(
        "--out_sot", requied=True, help="Target for SOT"
    )
    args = parser.parse_args(args)
    cfg = ConfigExtreme(args)
    recovery = common.Recovery(
        cfg.root_dir, args.config, cfg.fc_date, args.recover, shared_lock()
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
                param_name, cfg.fc_date, cfg.global_input_cfg, param_cfg, cfg.members
            )
            window_manager = common.WindowManager(param_cfg, cfg.global_output_cfg)
            efi_vars = ExtremeVariables(param_cfg)

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

            efi_partial = functools.partial(
                efi_sot, cfg, param, param_cfg["climatology"], efi_vars, recovery
            )
            for step, retrieved_data in parallel_data_retrieval(
                cfg.n_par_read,
                window_manager.unique_steps,
                [param],
                cfg.n_par_compute > 1,
            ):
                with common.ResourceMeter(f"Process step {step}"):
                    template, data = retrieved_data[0]

                    completed_windows = window_manager.update_windows(step, data)
                    for window_id, window in completed_windows:
                        executor.submit(efi_partial, template, window_id, window)

            executor.wait()

        recovery.clean_file()


if __name__ == "__main__":
    main()
