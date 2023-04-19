import eccodes

import pproc.common as common


def fdb_retrieve(step, *data_requesters):
    """
    Multiprocess retrieve data function from multiple data requests 
    with retrieve_data method. Grib message templates returned with 
    data are not returned.  

    :param step: integer step to retrieve data for
    :param data_requesters: list of objects with retrieve_data method
    accepting arguments (fdb, step)
    :return: tuple containing step and list of retrieved data
    """
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
