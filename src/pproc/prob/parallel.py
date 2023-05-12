import os
import pyfdb
import eccodes

from pproc import common
from pproc.prob.grib_helpers import construct_message
from pproc.prob.math import ensemble_probability


def write_grib(cfg, fdb, filename, template, data):
    output_file = os.path.join(cfg.options["root_dir"], filename)
    target = common.target_factory(cfg.options["target"], out_file=output_file, fdb=fdb)
    common.write_grib(target, template, data)


def prob_iteration(
    cfg,
    param,
    recovery,
    write_ensemble,
    template_filename,
    window_id,
    window,
    thresholds,
    additional_headers={},
):

    with common.ResourceMeter(f"Window {window.name}, computing threshold probs"):
        fdb = pyfdb.FDB()
        message_template = (
            template_filename
            if isinstance(template_filename, eccodes.highlevel.message.GRIBMessage)
            else common.io.read_template(template_filename)
        )

        if write_ensemble:
            for index in range(len(window.step_values)):
                data_type, number = param.type_and_number(index)
                print(
                    f"Writing window values for param {param.name} and output "
                    + f"type {data_type}, number {number} for step(s) {window.name}"
                )
                template = construct_message(message_template, window.grib_header())
                template.set({"type": data_type, "number": number})
                write_grib(
                    cfg,
                    fdb,
                    f"{param.name}_type{data_type}_number{number}_step{window.name}.grib",
                    template,
                    window.step_values[index],
                )
        for threshold in thresholds:
            window_probability = ensemble_probability(window.step_values, threshold)

            print(
                f"Writing probability for input param {param.name} and output "
                + f"param {threshold['out_paramid']} for step(s) {window.name}"
            )
            write_grib(
                cfg,
                fdb,
                f"{param.name}_{threshold['out_paramid']}_step{window.name}.grib",
                construct_message(
                    message_template,
                    window.grib_header(),
                    threshold,
                    additional_headers,
                ),
                window_probability,
            )

        fdb.flush()
        recovery.add_checkpoint(param.name, window_id)
