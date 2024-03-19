import eccodes
from meters import ResourceMeter

from pproc import common
from pproc.common.grib_helpers import construct_message
from pproc.prob.math import ensemble_probability


def prob_iteration(
    param,
    recovery,
    out_ensemble,
    out_prob,
    template_filename,
    window_id,
    accum,
    thresholds,
    additional_headers={},
):
    with ResourceMeter(f"Window {window_id}, computing threshold probs"):
        message_template = (
            template_filename
            if isinstance(template_filename, eccodes.highlevel.message.GRIBMessage)
            else common.io.read_template(template_filename)
        )

        ens = accum.values
        assert ens is not None

        if not isinstance(out_ensemble, common.io.NullTarget):
            for index in range(len(ens)):
                data_type, number = param.type_and_number(index)
                print(
                    f"Writing window values for param {param.name} and output "
                    + f"type {data_type}, number {number} for step(s) {window_id}"
                )
                template = construct_message(message_template, accum.grib_keys())
                template.set({"type": data_type, "number": number})
                common.write_grib(out_ensemble, template, ens[index])

        for threshold in thresholds:
            window_probability = ensemble_probability(ens, threshold)

            print(
                f"Writing probability for input param {param.name} and output "
                + f"param {threshold['out_paramid']} for step(s) {window_id}"
            )
            common.write_grib(
                out_prob,
                construct_message(
                    message_template,
                    accum.grib_keys(),
                    threshold,
                    additional_headers,
                ),
                window_probability,
            )

        out_ensemble.flush()
        out_prob.flush()
        recovery.add_checkpoint(param.name, window_id)
