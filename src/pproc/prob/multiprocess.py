import eccodes
import concurrent.futures as fut

import pproc.common as common


def fdb_retrieve(step, data_requesters, replace_grib: bool = False):
    """
    Retrieve data function for multiple data requests
    with retrieve_data method. If replace_grib is True then grib message
    templates returned with data are replace with None

    :param step: integer step to retrieve data for
    :param data_requesters: list of objects with retrieve_data method
    accepting arguments (fdb, step)
    :param replace_grib: boolean specifying whether to replace grib
    message templates with None
    :return: tuple containing step and list of retrieved data
    """
    with common.ResourceMeter(f"Retrieve step {step}"):
        collated_data = []
        fdb = common.io.fdb()
        for requester in data_requesters:
            template, data = requester.retrieve_data(fdb, step)
            if replace_grib and isinstance(
                template, eccodes.highlevel.message.GRIBMessage
            ):
                collated_data.append([None, data])
            else:
                collated_data.append([template, data])
        return [step, collated_data]


def parallel_data_retrieval(num_processes, steps, data_requesters, template_index=-1):
    """
    Multiprocess retrieve data function from multiple data requests
    with retrieve_data method.

    :param num_processes: number of processes to use for data retrieval
    :param steps: steps to retrieve data for
    :param data_requesters: list of objects with retrieve_data method
    accepting arguments (fdb, step)
    :param template_index: index to replace grib template for
    :return: iterator over list of step, list of retrieved data
    """
    if num_processes == 1:
        for step in steps:
            yield fdb_retrieve(step, data_requesters)
    else:
        if template_index >= 0:
            message_template, _ = data_requesters[template_index].retrieve_data(
                common.io.fdb(create=True), steps[0]
            )
        with fut.ProcessPoolExecutor(max_workers=num_processes) as pool:
            results = [
                pool.submit(fdb_retrieve, step, data_requesters, replace_grib=True)
                for step in steps
            ]
            for res in results:
                data_results = res.result()
                if template_index >= 0:
                    data_results[1][template_index][0] = message_template
                print(data_results)
                yield data_results
