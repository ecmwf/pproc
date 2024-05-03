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
from meters import ResourceMeter
from pproc import common
from pproc.common import parallel
from pproc.common.parallel import (
    SynchronousExecutor,
    QueueingExecutor,
    parallel_data_retrieval,
    sigterm_handler,
)


class ExtremeVariables:
    def __init__(self, efi_cfg):
        self.eps = float(efi_cfg["eps"])
        self.sot = list(map(int, efi_cfg["sot"]))
        self.compute_efi = efi_cfg.get("compute_efi", True)
        self.compute_sot = efi_cfg.get("compute_sot", True)
        self.compute_cpf = efi_cfg.get("compute_cpf", False)


def read_clim(fdb, climatology, window, n_clim=101, overrides={}):
    req = climatology["clim_keys"].copy()
    assert "date" in req
    req["time"] = "0000"
    req["quantile"] = ["{}:100".format(i) for i in range(n_clim)]
    if window.name in climatology.get("steps", {}):
        req["step"] = climatology["steps"][window.name]
    else:
        req["step"] = window.name
    req.update(overrides)

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
        "versionNumberOfExperimentalSuite",
        "implementationDateOfModelCycle",
        "numberOfReforecastYearsInModelClimate",
        "numberOfDaysInClimateSamplingWindow",
        "sampleSizeOfModelClimate",
        "versionOfModelClimate",
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


def cpf_template(template):
    template_cpf = template.copy()
    template_cpf["number"] = 0
    template_cpf["bitsPerValue"] = 24
    # TODO: add proper GRIB labelling once available
    return template_cpf


def efi_sot(
    cfg, param, climatology, efi_vars, recovery, template_filename, window_id, window
):
    with ResourceMeter(f"Window {window.suffix}, computing indices"):
        message_template = (
            template_filename
            if isinstance(template_filename, eccodes.highlevel.message.GRIBMessage)
            else common.io.read_template(template_filename)
        )

        clim, template_clim = read_clim(
            common.io.fdb(), climatology, window, overrides=cfg.override_input
        )
        print(f"Climatology array: {clim.shape}")

        template_extreme = extreme_template(window, message_template, template_clim)

        if efi_vars.compute_efi:
            control_index = param.get_type_index("cf", default=None)
            if control_index is not None:
                efi_control = extreme.efi(
                    clim, window.step_values[control_index], efi_vars.eps
                )
                template_efi = efi_template_control(template_extreme)
                common.write_grib(cfg.out_efi, template_efi, efi_control)

            efi = extreme.efi(clim, window.step_values, efi_vars.eps)
            template_efi = efi_template(template_extreme)
            common.write_grib(cfg.out_efi, template_efi, efi)

        if efi_vars.compute_sot:
            sot = {}
            for perc in efi_vars.sot:
                sot[perc] = extreme.sot(clim, window.step_values, perc, efi_vars.eps)
                template_sot = sot_template(template_extreme, perc)
                common.write_grib(cfg.out_sot, template_sot, sot[perc])

        if efi_vars.compute_cpf:
            cpf = 100 * extreme.cpf(clim.astype(np.float32), window.step_values.astype(np.float32), sort_clim=False, sort_ens=True)
            template_cpf = cpf_template(template_extreme)
            common.write_grib(cfg.out_cpf, template_cpf, cpf)

        cfg.out_efi.flush()
        cfg.out_sot.flush()
        recovery.add_checkpoint(param.name, window_id)


class ConfigExtreme(common.Config):
    def __init__(self, args):
        super().__init__(args)

        self.fc_date = datetime.strptime(str(self.options["fc_date"]), "%Y%m%d%H")

        if isinstance(self.options["members"], dict):
            self.members = range(
                self.options["members"]["start"], self.options["members"]["end"] + 1
            )
        else:
            self.members = int(self.options["members"])
        self.n_par_compute = self.options.get("n_par_compute", 1)
        self.n_par_read = self.options.get("n_par_read", 1)
        self.window_queue_size = self.options.get("queue_size", self.n_par_compute)

        self.root_dir = self.options["root_dir"]

        self.global_input_cfg = self.options.get("global_input_keys", {})
        self.global_output_cfg = self.options.get("global_output_keys", {})

        for attr in ["out_efi", "out_sot", "out_cpf"]:
            location = getattr(args, attr)
            target = common.io.target_from_location(
                location, overrides=self.override_output
            )
            if self.n_par_compute > 1:
                target.enable_parallel(parallel)
            if args.recover:
                target.enable_recovery()
            self.__setattr__(attr, target)

        print(f"Forecast date is {self.fc_date}")
        print(f"Root directory is {self.root_dir}")


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    parser = common.default_parser("Compute extreme indices from forecast and climatology")
    parser.add_argument("--out_efi", required=True, help="Target for EFI")
    parser.add_argument("--out_sot", required=True, help="Target for SOT")
    parser.add_argument("--out_cpf", default="null:", help="Target for CPF")
    args = parser.parse_args(args)
    cfg = ConfigExtreme(args)
    recovery = common.Recovery(
        cfg.root_dir, args.config, cfg.fc_date, args.recover
    )
    last_checkpoint = recovery.last_checkpoint()
    executor = (
        SynchronousExecutor()
        if cfg.n_par_compute == 1
        else QueueingExecutor(cfg.n_par_compute, cfg.window_queue_size, initializer=signal.signal,
                              initargs=(signal.SIGTERM, signal.SIG_DFL))
    )

    with executor:
        for param_name, param_cfg in sorted(cfg.options["parameters"].items()):
            param = common.create_parameter(
                param_name,
                cfg.fc_date,
                cfg.global_input_cfg,
                param_cfg,
                cfg.members,
                cfg.override_input,
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
                initializer=signal.signal,
                initargs=(signal.SIGTERM, signal.SIG_DFL)
            ):
                with ResourceMeter(f"Process step {step}"):
                    template, data = retrieved_data[0]
                    assert data.ndim == 2

                    completed_windows = window_manager.update_windows(step, data)
                    for window_id, window in completed_windows:
                        executor.submit(efi_partial, template, window_id, window)

            executor.wait()

        recovery.clean_file()


if __name__ == "__main__":
    main()
