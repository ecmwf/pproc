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
    window,
    thresholds,
    additional_headers={},
):

    with ResourceMeter(f"Window {window.name}, computing threshold probs"):
        message_template = (
            template_filename
            if isinstance(template_filename, eccodes.highlevel.message.GRIBMessage)
            else common.io.read_template(template_filename)
        )

        if not isinstance(out_ensemble, common.io.NullTarget):
            for index in range(len(window.step_values)):
                data_type, number = param.type_and_number(index)
                print(
                    f"Writing window values for param {param.name} and output "
                    + f"type {data_type}, number {number} for step(s) {window.name}"
                )
                template = construct_message(message_template, window.grib_header())
                template.set({"type": data_type, "number": number})
                common.write_grib(out_ensemble, template, window.step_values[index])

        for threshold in thresholds:
            window_probability = ensemble_probability(window.step_values, threshold)

            print(
                f"Writing probability for input param {param.name} and output "
                + f"param {threshold['out_paramid']} for step(s) {window.name}"
            )
            common.write_grib(
                out_prob,
                construct_message(
                    message_template,
                    window.grib_header(),
                    threshold,
                    additional_headers,
                ),
                window_probability,
            )

        out_ensemble.flush()
        out_prob.flush()
        recovery.add_checkpoint(param.name, window_id)
