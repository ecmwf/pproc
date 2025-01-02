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
from earthkit.meteo import extreme
from meters import ResourceMeter
from pproc import common
from pproc.common.grib_helpers import construct_message
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


def read_clim(fdb, climatology, accum, n_clim=101, overrides={}):
    grib_keys = accum.grib_keys()
    clim_step = grib_keys.get("stepRange", grib_keys.get("step", None))
    assert clim_step is not None

    req = climatology["clim_keys"].copy()
    assert "date" in req
    req["time"] = "0000"
    req["quantile"] = ["{}:100".format(i) for i in range(n_clim)]
    if clim_step in climatology.get("steps", {}):
        req["step"] = climatology["steps"][clim_step]
    else:
        req["step"] = clim_step
    req.update(overrides)

    print("Climatology request: ", req)
    da_clim = common.fdb_read(fdb, req)
    assert da_clim.values.shape[0] == n_clim
    da_clim_sorted = da_clim.reindex(quantile=[f"{x}:100" for x in range(n_clim)])
    print(da_clim_sorted)

    clim = np.asarray(da_clim_sorted.values)

    scale = climatology.get("scale", None)
    if scale is not None:
        clim *= float(scale)

    return clim, da_clim.attrs["grib_template"]


def extreme_template(accum, template_fc, template_clim):

    template_ext = construct_message(template_fc, accum.grib_keys())

    edition = template_ext["edition"]
    clim_edition = template_clim["edition"]
    if edition == 1 and clim_edition == 1:
        # EFI specific stuff
        if int(template_ext["timeRangeIndicator"]) == 3:
            if template_ext["numberIncludedInAverage"] == 0:
                template_ext["numberIncludedInAverage"] = len(accum)
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
    elif edition == 2 and clim_edition == 2:
        clim_keys = [
            "typeOfReferenceDataset",
            "yearOfStartOfReferencePeriod",
            "dayOfStartOfReferencePeriod",
            "monthOfStartOfReferencePeriod",
            "hourOfStartOfReferencePeriod",
            "minuteOfStartOfReferencePeriod",
            "secondOfStartOfReferencePeriod",
            "sampleSizeOfReferencePeriod",
            "numberOfReferencePeriodTimeRanges",
            "typeOfStatisticalProcessingForTimeRangeForReferencePeriod",
            "indicatorOfUnitForTimeRangeForReferencePeriod",
            "lengthOfTimeRangeForReferencePeriod",
        ]
        grib_keys = {
            "productDefinitionTemplateNumber": 105,
            **{key: template_clim[key] for key in clim_keys},
        }
        template_ext.set(grib_keys)
    else:
        raise Exception(
            f"Unsupported GRIB edition {edition} and clim edition {clim_edition}"
        )

    return template_ext


def efi_template(template):
    template_efi = template.copy()
    template_efi["marsType"] = 27

    edition = template_efi["edition"]
    if edition == 1:
        template_efi["efiOrder"] = 0
        template_efi["number"] = 0
    elif edition == 2:
        grib_set = {"typeOfRelationToReferenceDataset": 20, "typeOfProcessedData": 5}
        template_efi.set(grib_set)
    else:
        raise Exception(f"Unsupported GRIB edition {edition}")
    return template_efi


def efi_template_control(template):
    template_efi = template.copy()
    template_efi["marsType"] = 28

    edition = template_efi["edition"]
    if edition == 1:
        template_efi["efiOrder"] = 0
        template_efi["totalNumber"] = 1
        template_efi["number"] = 0
    elif edition == 2:
        grib_set = {"typeOfRelationToReferenceDataset": 20, "typeOfProcessedData": 3}
        template_efi.set(grib_set)
    else:
        raise Exception(f"Unsupported GRIB edition {edition}")
    return template_efi


def sot_template(template, sot):
    template_sot = template.copy()
    template_sot["marsType"] = 38

    if sot == 90:
        efi_order = 99
    elif sot == 10:
        efi_order = 1
    else:
        raise Exception(
            f"SOT value '{sot}' not supported in template! Only accepting 10 and 90"
        )
    edition = template_sot["edition"]
    if edition == 1:
        template_sot["number"] = sot
        template_sot["efiOrder"] = efi_order
    elif edition == 2:
        grib_set = {
            "typeOfRelationToReferenceDataset": 21,
            "typeOfProcessedData": 5,
            "numberOfAdditionalParametersForReferencePeriod": 2,
            "scaleFactorOfAdditionalParameterForReferencePeriod": [0, 0],
            "scaledValueOfAdditionalParameterForReferencePeriod": [sot, efi_order],
        }
        template_sot.set(grib_set)
    else:
        raise Exception(f"Unsupported GRIB edition {edition}")
    return template_sot


def cpf_template(template):
    template_cpf = template.copy()
    template_cpf["number"] = 0
    template_cpf["bitsPerValue"] = 24
    # TODO: add proper GRIB labelling once available
    return template_cpf


def efi_sot(
    cfg, param, climatology, efi_vars, recovery, template_filename, window_id, accum
):
    with ResourceMeter(f"Window {window_id}, computing indices"):
        message_template = (
            template_filename
            if isinstance(template_filename, eccodes.highlevel.message.GRIBMessage)
            else common.io.read_template(template_filename)
        )

        clim, template_clim = read_clim(
            common.io.fdb(),
            climatology,
            accum,
            overrides=cfg.override_input,
        )
        print(f"Climatology array: {clim.shape}")

        template_extreme = extreme_template(accum, message_template, template_clim)

        ens = accum.values
        assert ens is not None

        if efi_vars.compute_efi:
            ens_types = param.base_request["type"].split("/")
            if "cf" in ens_types or "fc" in ens_types:
                efi_control = extreme.efi(
                    clim, ens.sel(number=[0]).values, efi_vars.eps
                )
                template_efi = efi_template_control(template_extreme)
                common.write_grib(cfg.out_efi, template_efi, efi_control)

            efi = extreme.efi(clim, ens.values, efi_vars.eps)
            template_efi = efi_template(template_extreme)
            common.write_grib(cfg.out_efi, template_efi, efi)
            cfg.out_efi.flush()

        if efi_vars.compute_sot:
            sot = {}
            for perc in efi_vars.sot:
                sot[perc] = extreme.sot(clim, ens.values, perc, efi_vars.eps)
                template_sot = sot_template(template_extreme, perc)
                common.write_grib(cfg.out_sot, template_sot, sot[perc])
            cfg.out_sot.flush()

        if efi_vars.compute_cpf:
            cpf = 100 * extreme.cpf(
                clim.astype(np.float32),
                ens.values.astype(np.float32),
                sort_clim=False,
                sort_ens=True,
            )
            template_cpf = cpf_template(template_extreme)
            common.write_grib(cfg.out_cpf, template_cpf, cpf)
            cfg.out_cpf.flush()

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

    parser = common.default_parser(
        "Compute extreme indices from forecast and climatology"
    )
    parser.add_argument("--out_efi", required=True, help="Target for EFI")
    parser.add_argument("--out_sot", required=True, help="Target for SOT")
    parser.add_argument("--out_cpf", default="null:", help="Target for CPF")
    args = parser.parse_args(args)
    cfg = ConfigExtreme(args)
    recovery = common.Recovery(cfg.root_dir, args.config, cfg.fc_date, args.recover)
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
                new_start = window_manager.delete_windows(checkpointed_windows)
                print(f"Recovery: param {param_name} looping from step {new_start}")
                last_checkpoint = None  # All remaining params have not been run

            efi_partial = functools.partial(
                efi_sot, cfg, param, param_cfg["climatology"], efi_vars, recovery
            )
            for keys, retrieved_data in parallel_data_retrieval(
                cfg.n_par_read,
                window_manager.dims,
                [param],
                cfg.n_par_compute > 1,
                initializer=signal.signal,
                initargs=(signal.SIGTERM, signal.SIG_DFL),
            ):
                step = keys["step"]
                with ResourceMeter(f"Process step {step}"):
                    template, data = retrieved_data[0]
                    assert data.ndim == 2

                    completed_windows = window_manager.update_windows(keys, data)
                    for window_id, accum in completed_windows:
                        executor.submit(efi_partial, template, window_id, accum)

            executor.wait()

        recovery.clean_file()


if __name__ == "__main__":
    main()
