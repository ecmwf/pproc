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


import sys
from datetime import datetime
import functools
import signal
from typing import Dict, Any

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
from pproc.common.param_requester import ParamConfig, ParamRequester, index_ensembles
from pproc.signi.clim import retrieve_clim


class ExtremeParamConfig(ParamConfig):
    def __init__(
        self, name: str, options: Dict[str, Any], overrides: Dict[str, Any] = {}
    ):
        super().__init__(name, options, overrides)
        clim_options = options.copy()
        if "clim" in options:
            clim_options.update(clim_options.pop("clim"))
        self.clim_param = ParamConfig(f"clim_{name}", clim_options)
        self.eps = float(options["eps"])
        self.sot = list(map(int, options["sot"]))


class ConfigExtreme(common.Config):
    def __init__(self, args):
        super().__init__(args)

        self.fc_date = datetime.strptime(str(self.options["fc_date"]), "%Y%m%d%H")
        self.members = self.options.get("num_members", 51)
        self.total_fields = self.options.get("total_fields", self.members)
        self.n_par_compute = self.options.get("n_par_compute", 1)
        self.n_par_read = self.options.get("n_par_read", 1)
        self.window_queue_size = self.options.get("queue_size", self.n_par_compute)

        self.root_dir = self.options["root_dir"]
        self.out_keys = self.options.get("out_keys", {})

        self.sources = self.options.get("sources", {})

        self.steps = self.options.get("steps", [])
        self.windows = self.options.get("windows", [])

        self.parameters = [
            ExtremeParamConfig(pname, popt, overrides=self.override_input)
            for pname, popt in self.options["parameters"].items()
        ]
        self.clim_loc = args.in_clim

        for attr in ["out_efi", "out_sot"]:
            location = getattr(args, attr)
            target = common.io.target_from_location(
                location, overrides=self.override_output
            )
            if self.n_par_compute > 1:
                target.enable_parallel(parallel)
            if args.recover:
                target.enable_recovery()
            self.__setattr__(attr, target)


def read_clim(config: ConfigExtreme, param: ExtremeParamConfig, accum, n_clim=101):
    grib_keys = accum.grib_keys()
    clim_step = grib_keys.get("stepRange", grib_keys.get("step", None))
    in_keys = param.clim_param._in_keys
    in_keys["quantile"] = ["{}:100".format(i) for i in range(n_clim)]
    step = in_keys.get("step", {}).get(clim_step, clim_step)
    clim_accum, clim_template = retrieve_clim(
        param.clim_param,
        config.sources,
        config.clim_loc,
        1,
        n_clim,
        index_func=lambda x: int(x.get("quantile").split(":")[0]),
        step=step,
    )
    if not isinstance(clim_template, eccodes.GRIBMessage):
        clim_template = common.io.read_template(clim_template)
    return clim_accum.values, clim_template


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


def efi_sot(cfg, param, recovery, template_filename, window_id, accum):
    with ResourceMeter(f"Window {window_id}, computing EFI/SOT"):
        message_template = (
            template_filename
            if isinstance(template_filename, eccodes.highlevel.message.GRIBMessage)
            else common.io.read_template(template_filename)
        )

        clim, template_clim = read_clim(cfg, param, accum)
        print(f"Climatology array: {clim.shape}")

        template_extreme = extreme_template(accum, message_template, template_clim)

        ens = accum.values
        assert ens is not None

        if message_template.get("type") in ["cf", "fc"]:
            efi_control = extreme.efi(clim, ens[:1, :], param.eps)
            template_efi = efi_template_control(template_extreme)
            common.write_grib(cfg.out_efi, template_efi, efi_control)

        efi = extreme.efi(clim, ens, param.eps)
        template_efi = efi_template(template_extreme)
        common.write_grib(cfg.out_efi, template_efi, efi)

        sot = {}
        for perc in param.sot:
            sot[perc] = extreme.sot(clim, ens, perc, param.eps)
            template_sot = sot_template(template_extreme, perc)
            common.write_grib(cfg.out_sot, template_sot, sot[perc])

        cfg.out_efi.flush()
        cfg.out_sot.flush()
        recovery.add_checkpoint(param.name, window_id)


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    parser = common.default_parser("Compute EFI and SOT from forecast and climatology")
    parser.add_argument("--in-ens", required=True, help="Source for forecast")
    parser.add_argument("--in-clim", required=True, help="Source for climatology")
    parser.add_argument("--out-efi", required=True, help="Target for EFI")
    parser.add_argument("--out-sot", required=True, help="Target for SOT")
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
        for param in cfg.parameters:
            requester = ParamRequester(
                param,
                cfg.sources,
                args.in_ens,
                cfg.members,
                cfg.total_fields,
                index_ensembles,
            )
            window_manager = common.WindowManager(
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

            efi_partial = functools.partial(efi_sot, cfg, param, recovery)
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
                    template, data = retrieved_data[0]
                    assert data.ndim == 2

                    completed_windows = window_manager.update_windows(keys, data)
                    for window_id, accum in completed_windows:
                        executor.submit(efi_partial, template, window_id, accum)

            executor.wait()

        recovery.clean_file()


if __name__ == "__main__":
    main()
