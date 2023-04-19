import pproc.common as common

DEFAULT_NUM_PROCESSES = 2

def retrieve(step, *data_requesters):
    with common.ResourceMeter(f"Retrieve step {step}"):
        collated_data = []
        fdb = common.io.fdb()
        for requester in data_requesters:
            collated_data.append(requester.retrieve_data(fdb, step))
        return (step, collated_data)