import eccodes

import pproc.common as common


def retrieve(step, *data_requesters):
    with common.ResourceMeter(f"Retrieve step {step}"):
        collated_data = []
        fdb = common.io.fdb()
        for requester in data_requesters:
            template, data = requester.retrieve_data(fdb, step)
            if isinstance(template, eccodes.highlevel.message.GRIBMessage):
                collated_data.append((None, data))
            else:
                collated_data.append((template, data))
        return (step, collated_data)
