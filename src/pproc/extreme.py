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


import os
import numpy as np
import sys
from datetime import datetime
import functools
import concurrent.futures as fut

import pyfdb
from meteokit import extreme
from pproc import common
from pproc.common.parallel import create_executor


class ExtremeVariables:
    def __init__(self, efi_cfg):
        self.eps = float(efi_cfg["eps"])
        self.sot = list(map(int, efi_cfg["sot"]))


def read_clim(cfg, clim_keys, window, n_clim=101):

    req = clim_keys.copy()
    req["date"] = cfg.clim_date.strftime("%Y%m%d")
    req["time"] = "0000"
    req["quantile"] = ["{}:100".format(i) for i in range(n_clim)]
    req["step"] = f"{window.name}"

    print("Climatology request: ", req)
    da_clim = common.fdb_read(cfg.fdb, req)
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


def efi_sot(param, clim, window, efi_vars):
    with common.ResourceMeter(f"Window {window.suffix}, computing EFI/SOT"):
        results = {
            "efi_control": extreme.efi(
                clim, window.step_values[param.get_type_index("cf")], efi_vars.eps
            ),
            "efi": extreme.efi(clim, window.step_values, efi_vars.eps),
        }
        for perc in efi_vars.sot:
            results[f"sot_{perc}"] = extreme.sot(
                clim, window.step_values, perc, efi_vars.eps
            )

        return results


def write_outputs(cfg, param_name, window, template, recovery, future):
    for computation_type, computed_values in future.result().items():
        if computation_type == "efi_control":
            template_efi = efi_template_control(template)
        elif computation_type == "efi":
            template_efi = efi_template(template)
        elif "sot" in computation_type:
            perc = int(computation_type.split("_")[1])
            template_efi = sot_template(template, perc)

        out_file = os.path.join(
            cfg.out_dir, f"{computation_type}_{param_name}_{window.suffix}.grib"
        )
        target = common.target_factory(cfg.target, out_file=out_file, fdb=cfg.fdb)
        common.write_grib(target, template_efi, computed_values)
        cfg.fdb.flush()
        recovery.add_checkpoint(param_name, window.name)


class ConfigExtreme(common.Config):
    def __init__(self, args):
        super().__init__(args)

        self.fc_date = datetime.strptime(str(self.options["fc_date"]), "%Y%m%d%H")

        self.members = int(self.options["members"])
        self.fdb = pyfdb.FDB()
        self.n_par = self.options.get("n_par", 1)

        self.root_dir = self.options["root_dir"]
        self.out_dir = os.path.join(
            self.root_dir, "efi_test", self.fc_date.strftime("%Y%m%d%H")
        )

        self.clim_date = self.options.get(
            "clim_date", common.climatology_date(self.fc_date)
        )

        self.target = self.options["target"]
        self.global_input_cfg = self.options.get("global_input_keys", {})
        self.global_output_cfg = self.options.get("global_output_keys", {})

        print(f"Forecast date is {self.fc_date}")
        print(f"Climatology date is {self.clim_date}")
        print(f"Root directory is {self.root_dir}")


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)

    parser = common.default_parser(
        "Compute EFI and SOT from forecast and climatology for one parameter"
    )
    args = parser.parse_args(args)
    cfg = ConfigExtreme(args)
    recovery = common.Recovery(cfg.root_dir, args.config, cfg.fc_date, args.recover)
    last_checkpoint = recovery.last_checkpoint()
    executor = create_executor(cfg.n_par)

    for param_name, param_cfg in sorted(cfg.options["parameters"].items()):
        param = common.create_parameter(
            cfg.fc_date, cfg.global_input_cfg, param_cfg, cfg.members
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

        all_futures = []
        for step in window_manager.unique_steps:
            with common.ResourceMeter(f"Parameter {param_name}, step {step}"):
                message_template, data = param.retrieve_data(cfg.fdb, step)

                completed_windows = window_manager.update_windows(step, data)
                for window in completed_windows:

                    clim, template_clim = read_clim(cfg, param_cfg["clim_keys"], window)
                    print(f"Climatology array: {clim.shape}")

                    window_future = executor.submit(
                        efi_sot, param, clim, window, efi_vars
                    )

                    template_extreme = extreme_template(
                        window, message_template, template_clim
                    )
                    write_callback = functools.partial(
                        write_outputs,
                        cfg,
                        param_name,
                        window,
                        template_extreme,
                        recovery,
                    )
                    window_future.add_done_callback(write_callback)

                    all_futures.append(window_future)

        fut.wait(all_futures)

    recovery.clean_file()


if __name__ == "__main__":
    main()
