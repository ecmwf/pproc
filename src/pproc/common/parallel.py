import concurrent.futures as fut
from typing import List
import eccodes

from pproc.common import Parameter, ResourceMeter, io


class SynchronousExecutor(fut.Executor):
    """Dummy Executor that calls the functions directly"""

    def submit(self, fn, *args, **kwargs):
        f = fut.Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except Exception as e:
            f.set_exception(e)
        return f

    def wait(self):
        pass


class QueueingExecutor(fut.ProcessPoolExecutor):
    """
    Executor with queue for pending futures. Blocks submission
    of new jobs until number of pending futures is below
    required limit. Useful for controlling memory usage if the data
    required for pending futures can be large.
    """

    def __init__(self, n_par: int, queue_size: int = 0):
        """
        :param n_par: number of processes
        :queue_size: maximum number of allowed pending futures, if 0 then
        no queueing is implemented
        """
        super().__init__(max_workers=n_par)
        self.futures = []
        self.queue_size = queue_size

    def submit(self, function, *args, **kwargs):
        """
        Submission of new jobs is blocked if the number of pending
        futures is larger than maximum allowed. Waits for first future
        completion, removes all complete futures and then submits
        new job
        """
        if self.queue_size > 0 and len(self.futures) >= self.queue_size:
            print(
                f"Queue reached max limit {self.queue_size}. Waiting for a subprocess completion"
            )
            fut.wait(self.futures, return_when="FIRST_COMPLETED")
            new_futures = []
            for future in self.futures:
                if future.done():
                    future.result()
                else:
                    new_futures.append(future)
            self.futures[:] = new_futures

        self.futures.append(super().submit(function, *args, **kwargs))

    def wait(self):
        """
        Wait for futures to complete and fetch results
        """
        for future in fut.as_completed(self.futures):
            future.result()


def parallel_processing(process, plan, n_par, recovery=None):
    """Run a processing function in parallel

    Parameters
    ----------
    process: callable
        Processing function to call. The return value is used as a recovery key.
    plan: iterable of tuples
        Arguments for the processing function
    n_par: int
        Number of parallel processes
    recovery: Recovery or None
        If set, add checkpoints when processing succeeds
    """
    executor = (
        SynchronousExecutor()
        if n_par == 1
        else fut.ProcessPoolExecutor(max_workers=n_par)
    )
    with executor:
        for future in fut.as_completed(
            executor.submit(process, *args) for args in plan
        ):
            key = future.result()
            if recovery is not None:
                recovery.add_checkpoint(*key)


def fdb_retrieve(
    step: int, data_requesters: List[Parameter], grib_to_file: bool = False
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
    with ResourceMeter(f"Retrieve step {step}"):
        collated_data = []
        fdb = io.fdb()
        for requester in data_requesters:
            template, data = requester.retrieve_data(fdb, step)
            if grib_to_file and isinstance(
                template, eccodes.highlevel.message.GRIBMessage
            ):
                filename = f"template_{requester.name}_step{step}.grib"
                io.write_template(filename, template)
                collated_data.append([filename, data])
            else:
                collated_data.append([template, data])
        return collated_data


def parallel_data_retrieval(
    num_processes: int, steps: List[int], data_requesters: List[Parameter], grib_to_file: bool = False
):
    """
    Multiprocess retrieve data function from multiple data requests
    with retrieve_data method. If grib_to_file is true then message templates from the fdb requests are
    written to file and the filename returned with the data, else the message template itself is returned.

    :param num_processes: number of processes to use for data retrieval
    :param steps: steps to retrieve data for
    :param data_requesters: list of Parameter instances
    :return: iterator over step, retrieved data
    """
    if num_processes == 1:
        for step in steps:
            yield step, fdb_retrieve(step, data_requesters, grib_to_file)
    else:
        with fut.ProcessPoolExecutor(max_workers=num_processes) as executor:
            n_initial_submit = min(num_processes, len(steps))
            futures = [
                executor.submit(
                    fdb_retrieve, steps[submit_index], data_requesters, True
                )
                for submit_index in range(n_initial_submit)
            ]
            for step_index, step in enumerate(steps):
                # Submit as steps get processed to avoid out of memory problems
                submit_index = step_index + n_initial_submit
                if submit_index < len(steps):
                    futures.append(
                        executor.submit(
                            fdb_retrieve, steps[submit_index], data_requesters, True
                        )
                    )

                # Steps need to be processed in order so block until data for next step
                # is available
                data_results = futures.pop(0).result()
                if not grib_to_file:
                    for result_index, result in enumerate(data_results):
                        if isinstance(result[0], str):
                            data_results[result_index][0] = io.read_template(result[0])
                yield step, data_results
