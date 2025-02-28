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
import functools
import numpy as np

import eccodes
from earthkit.meteo import extreme
from meters import ResourceMeter
from conflator import Conflator

from pproc import common
from pproc.common.accumulation import Accumulator
from pproc.common.grib_helpers import construct_message
from pproc.common.window_manager import WindowManager
from pproc.common.recovery import create_recovery, Recovery
from pproc.common.parallel import create_executor, parallel_data_retrieval
from pproc.common.param_requester import ParamRequester
from pproc.config.types import ExtremeParamConfig, ExtremeConfig
from pproc.signi.clim import retrieve_clim


def read_clim(
    config: ExtremeConfig,
    param: ExtremeParamConfig,
    accum: Accumulator,
    n_clim: int = 101,
) -> tuple[np.ndarray, eccodes.GRIBMessage]:
    grib_keys = accum.grib_keys()
    clim_step = grib_keys.get("stepRange", grib_keys.get("step", None))
    clim_request = param.sources["clim"]["request"]
    clim_request["quantile"] = ["{}:100".format(i) for i in range(n_clim)]
    step = clim_request.get("step", {}).get(clim_step, clim_step)
    clim_accum, clim_template = retrieve_clim(
        param,
        config.sources,
        "clim",
        1,
        n_clim,
        index_func=lambda x: int(x.get("quantile").split(":")[0]),
        step=step,
    )
    if not isinstance(clim_template, eccodes.GRIBMessage):
        clim_template = common.io.read_template(clim_template)
    return clim_accum.values, clim_template


def extreme_template(
    accum: Accumulator,
    template_fc: eccodes.GRIBMessage,
    template_clim: eccodes.GRIBMessage,
) -> eccodes.GRIBMessage:

    template_ext = construct_message(template_fc, accum.grib_keys())
    grib_keys = {}

    edition = template_ext["edition"]
    clim_edition = template_clim["edition"]
    if edition == 1 and clim_edition == 1:
        # EFI specific stuff
        if int(template_ext["timeRangeIndicator"]) == 3:
            if template_ext["numberIncludedInAverage"] == 0:
                grib_keys["numberIncludedInAverage"] = len(accum)
            grib_keys["numberMissingFromAveragesOrAccumulations"] = 0

        # set clim keys
        clim_keys = [
            "versionNumberOfExperimentalSuite",
            "implementationDateOfModelCycle",
            "numberOfReforecastYearsInModelClimate",
            "numberOfDaysInClimateSamplingWindow",
            "sampleSizeOfModelClimate",
            "versionOfModelClimate",
        ]
        for key in clim_keys:
            grib_keys[key] = template_clim[key]

        # set fc keys
        fc_keys = [
            "date",
            "subCentre",
            "totalNumber",
        ]
        for key in fc_keys:
            grib_keys[key] = template_fc[key]
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
        grib_keys.update(
            {
                "productDefinitionTemplateNumber": 105,
                **{key: template_clim[key] for key in clim_keys},
            }
        )
    else:
        raise Exception(
            f"Unsupported GRIB edition {edition} and clim edition {clim_edition}"
        )

    template_ext.set(grib_keys)
    return template_ext


def efi_template(template: eccodes.GRIBMessage) -> eccodes.GRIBMessage:
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


def efi_template_control(template: eccodes.GRIBMessage) -> eccodes.GRIBMessage:
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


def sot_template(template: eccodes.GRIBMessage, sot: float) -> eccodes.GRIBMessage:
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


def efi_sot(
    cfg: ExtremeConfig,
    param: ExtremeParamConfig,
    recovery: Recovery,
    template_filename: str,
    window_id: str,
    accum: Accumulator,
):
    with ResourceMeter(f"Window {window_id}, computing EFI/SOT"):
        message_template = (
            template_filename
            if isinstance(template_filename, eccodes.highlevel.message.GRIBMessage)
            else common.io.read_template(template_filename)
        )

        clim, template_clim = read_clim(cfg, param.clim, accum)
        print(f"Climatology array: {clim.shape}")

        template_extreme = extreme_template(accum, message_template, template_clim)

        ens = accum.values
        assert ens is not None

        if message_template.get("type") in ["cf", "fc"]:
            efi_control = extreme.efi(clim, ens[:1, :], param.eps)
            template_efi = efi_template_control(template_extreme)
            common.io.write_grib(cfg.outputs.efi.target, template_efi, efi_control)

        efi = extreme.efi(clim, ens, param.eps)
        template_efi = efi_template(template_extreme)
        common.io.write_grib(cfg.outputs.efi.target, template_efi, efi)
        cfg.outputs.efi.target.flush()

        sot = {}
        for perc in param.sot:
            sot[perc] = extreme.sot(clim, ens, perc, param.eps)
            template_sot = sot_template(template_extreme, perc)
            common.io.write_grib(cfg.outputs.sot.target, template_sot, sot[perc])
        cfg.outputs.sot.target.flush()

        recovery.add_checkpoint(param=param.name, window=window_id)


def main():
    sys.stdout.reconfigure(line_buffering=True)

    cfg = Conflator(app_name="pproc-extreme", model=ExtremeConfig).load()
    cfg.print()
    recovery = create_recovery(cfg)

    with create_executor(cfg.parallelisation) as executor:
        for param in cfg.parameters:
            print(f"Processing {param.name}")
            window_manager = WindowManager(
                param.accumulations,
                {
                    **cfg.outputs.default.metadata,
                    **param.metadata,
                },
            )
            checkpointed_windows = [
                x["window"] for x in recovery.computed(param=param.name)
            ]
            new_start = window_manager.delete_windows(checkpointed_windows)
            if new_start is None:
                print(f"Recovery: skipping completed param {param.name}")
                continue

            print(f"Recovery: param {param.name} starting from step {new_start}")

            requester = ParamRequester(
                param, cfg.sources, cfg.members, cfg.total_fields, "fc"
            )
            efi_partial = functools.partial(efi_sot, cfg, param, recovery)
            for keys, data in parallel_data_retrieval(
                cfg.parallelisation.n_par_read,
                window_manager.dims,
                [requester],
                cfg.parallelisation.n_par_compute > 1,
            ):
                step = keys["step"]
                with ResourceMeter(f"Process step {step}"):
                    template, ens = data[0]
                    assert ens.ndim == 2

                    completed_windows = window_manager.update_windows(keys, ens)
                    for window_id, accum in completed_windows:
                        executor.submit(efi_partial, template, window_id, accum)

            executor.wait()
    recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
