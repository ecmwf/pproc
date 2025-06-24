# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime, timedelta

from pproc.common.grib_helpers import construct_message


def extreme_template(accum, template_fc, template_clim, allow_grib1_to_grib2=False):

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
    elif edition == 2 and clim_edition == 1 and allow_grib1_to_grib2:
        # WARNING: this is highly experimental
        stat_keys = [
            "typeOfStatisticalProcessing",
            "typeOfTimeIncrement",
            "indicatorOfUnitForTimeRange",
            "lengthOfTimeRange",
            "indicatorOfUnitForTimeIncrement",
            "timeIncrement",
        ]
        in_stat = {key: template_ext[key] for key in stat_keys}
        ext_stat = in_stat.copy()
        ext_stat["typeOfStatisticalProcessing"] = 102
        if in_stat["typeOfStatisticalProcessing"] == 0:
            set_stat = [ext_stat]
        else:
            ext_stat["indicatorOfUnitForTimeIncrement"] = 255
            ext_stat["timeIncrement"] = 0
            set_stat = [ext_stat, in_stat]
        grib_keys.update(
            {
                "typeOfProcessedData": 255,
                "productDefinitionTemplateNumber": 107,
                "numberOfTimeRanges": len(set_stat),
            }
        )
        template_ext.set(grib_keys)
        for key in ext_stat.keys():
            template_ext.set_array(key, [st[key] for st in set_stat])
        grib_keys = {}
        grib_keys.update(
            {
                "derivedForecast": 255,
            }
        )
        clim_date = datetime.strptime(template_clim["date:str"], "%Y%m%d")
        clim_nyears = template_clim["numberOfReforecastYearsInModelClimate"]
        clim_start = clim_date.replace(year=clim_date.year - clim_nyears)
        clim_size = template_clim["sampleSizeOfModelClimate"]
        clim_window = template_clim["numberOfDaysInClimateSamplingWindow:int"]
        clim_start -= timedelta(days=clim_window // 2)
        grib_keys.update(
            {
                "typeOfReferenceDataset": 2,
                "yearOfStartOfReferencePeriod": clim_start.year,
                "monthOfStartOfReferencePeriod": clim_start.month,
                "dayOfStartOfReferencePeriod": clim_start.day,
                "hourOfStartOfReferencePeriod": 0,
                "minuteOfStartOfReferencePeriod": 0,
                "secondOfStartOfReferencePeriod": 0,
                "sampleSizeOfReferencePeriod": clim_size,
                "numberOfReferencePeriodTimeRanges": 2,
            }
        )
        template_ext.set(grib_keys)
        arr_grib_keys = {
            "typeOfStatisticalProcessingForTimeRangeForReferencePeriod": [20, 20],
            "indicatorOfUnitForTimeRangeForReferencePeriod": [4, 2],
            "lengthOfTimeRangeForReferencePeriod": [clim_nyears, clim_window],
        }
        for key, value in arr_grib_keys.items():
            template_ext.set_array(key, value)
        grib_keys = {}
    else:
        raise Exception(
            f"Unsupported GRIB edition {edition} and clim edition {clim_edition}"
        )

    template_ext.set(grib_keys)
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
    template_cpf[
        "marsType"
    ] = 27  # FIXME: this corresponds to efi, should be a new value for cpf
    template_cpf["bitsPerValue"] = 24

    edition = template_cpf["edition"]
    if edition == 1:
        template_cpf["number"] = 0
    elif edition == 2:
        grib_set = {"typeOfRelationToReferenceDataset": 24, "typeOfProcessedData": 5}
        template_cpf.set(grib_set)
    else:
        raise Exception(f"Unsupported GRIB edition {edition}")
    return template_cpf
