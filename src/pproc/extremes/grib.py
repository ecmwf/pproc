from pproc.common.grib_helpers import construct_message


def extreme_template(accum, template_fc, template_clim):

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
