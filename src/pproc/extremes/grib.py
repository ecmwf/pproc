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
        ]
        for key in fc_keys:
            grib_keys[key] = template_fc[key]
        total_number = template_fc.get("totalNumber", len(accum.values), int)
        grib_keys["totalNumber"] = (
            len(accum.values) if total_number == 0 else total_number
        )
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
                "productDefinitionTemplateNumber": 107,
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

    return template_ext, grib_keys


def efi_metadata(template, metadata) -> dict:
    metadata = metadata.copy()
    metadata["marsType"] = 27

    edition = metadata.get("edition", template["edition"])
    if edition == 1:
        metadata["efiOrder"] = 0
        metadata["number"] = 0
    elif edition == 2:
        metadata["typeOfRelationToReferenceDataset"] = 20
        metadata.setdefault("typeOfProcessedData", 5)
    else:
        raise Exception(f"Unsupported GRIB edition {edition}")
    return metadata


def efi_metadata_control(template, metadata) -> dict:
    metadata = metadata.copy()
    metadata["marsType"] = 28

    edition = metadata.get("edition", template["edition"])
    if edition == 1:
        metadata["efiOrder"] = 0
        metadata["totalNumber"] = 1
        metadata["number"] = 0
    elif edition == 2:
        metadata["typeOfRelationToReferenceDataset"] = 20
        metadata.setdefault("typeOfProcessedData", 3)
    else:
        raise Exception(f"Unsupported GRIB edition {edition}")
    return metadata


def sot_metadata(template, sot, metadata) -> dict:
    metadata = metadata.copy()

    if sot == 90:
        efi_order = 99
    elif sot == 10:
        efi_order = 1
    else:
        raise Exception(
            f"SOT value '{sot}' not supported in template! Only accepting 10 and 90"
        )
    edition = metadata.get("edition", template["edition"])
    if edition == 1:
        metadata["marsType"] = 38
        metadata["number"] = sot
        metadata["efiOrder"] = efi_order
    elif edition == 2:
        metadata.setdefault("typeOfProcessedData", 5)
        metadata.update(
            {
                "typeOfRelationToReferenceDataset": 21,
                "numberOfAdditionalParametersForReferencePeriod": 2,
                "scaleFactorOfAdditionalParameterForReferencePeriod": [0, 0],
                "scaledValueOfAdditionalParameterForReferencePeriod": [sot, efi_order],
                "marsType": 38,
            }
        )
    else:
        raise Exception(f"Unsupported GRIB edition {edition}")
    return metadata


def cpf_metadata(template, metadata) -> dict:
    metadata = metadata.copy()
    metadata["marsType"] = (
        27  # FIXME: this corresponds to efi, should be a new value for cpf
    )
    metadata["bitsPerValue"] = 24

    edition = metadata.get("edition", template["edition"])
    if edition == 1:
        metadata["number"] = 0
    elif edition == 2:
        metadata["typeOfRelationToReferenceDataset"] = 24
        metadata.setdefault("typeOfProcessedData", 5)
    else:
        raise Exception(f"Unsupported GRIB edition {edition}")
    return metadata
