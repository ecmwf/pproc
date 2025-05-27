import eccodes

from pproc.common.grib_helpers import construct_message


def quantiles_template(
    template: eccodes.GRIBMessage,
    pert_number: int,
    total_number: int,
    out_keys: dict,
) -> eccodes.GRIBMessage:

    edition = out_keys.get("edition", template.get("edition"))
    if edition not in (1, 2):
        raise ValueError(f"Unsupported GRIB edition {edition}")
    grib_keys = {**out_keys}
    if edition == 1:
        grib_keys.update(
            {
                "totalNumber": total_number,
                "perturbationNumber": pert_number,
            }
        )
    else:
        grib_keys.setdefault("productDefinitionTemplateNumber", 86)
        grib_keys.update(
            {
                "totalNumberOfQuantiles": total_number,
                "quantileValue": pert_number,
            }
        )
    message = construct_message(template, grib_keys)
    return message
