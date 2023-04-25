import eccodes
from typing import List
import concurrent.futures as fut

import pproc.common as common


def fdb_retrieve(
    step: int, data_requesters: List[common.Parameter], grib_to_file: bool = False
):
    """
    Retrieve data function for multiple data requests
    with retrieve_data method. If requested, grib template messages are written to
    file and their filename returned

    :param step: integer step to retrieve data for
    :param data_requesters: list of objects with retrieve_data method
    accepting arguments (fdb, step)
    :param grib_to_file: boolean specifying whether to write grib messages to file and return filename
    :return: tuple containing step and list of retrieved data
    """
    with common.ResourceMeter(f"Retrieve step {step}"):
        collated_data = []
        fdb = common.io.fdb()
        for requester in data_requesters:
            template, data = requester.retrieve_data(fdb, step)
            if grib_to_file and isinstance(
                template, eccodes.highlevel.message.GRIBMessage
            ):
                filename = f"template_{requester.name}_step{step}.grib"
                common.io.write_template(filename, template)
                collated_data.append([filename, data])
            else:
                collated_data.append([template, data])
        return collated_data


def parallel_data_retrieval(
    num_processes: int, steps: List[int], data_requesters: List[common.Parameter]
):
    """
    Multiprocess retrieve data function from multiple data requests
    with retrieve_data method.

    :param num_processes: number of processes to use for data retrieval
    :param steps: steps to retrieve data for
    :param data_requesters: list of Parameter instances
    :return: iterator over step, retrieved data
    """
    if num_processes == 1:
        for step in steps:
            yield step, fdb_retrieve(step, data_requesters)
    else:
        with fut.ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(fdb_retrieve, step, data_requesters, True)
                for step in steps
            ]
            for step_index, future in enumerate(futures):
                # Steps need to be processed in order so block until data for next step
                # is available
                data_results = future.result()
                for result_index, result in enumerate(data_results):
                    if isinstance(result[0], str):
                        data_results[result_index][0] = common.io.read_template(result[0])
                yield steps[step_index], data_results
