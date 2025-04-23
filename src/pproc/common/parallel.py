import concurrent.futures as fut
import os
import sys
from typing import List
import signal

import psutil
from meters import ResourceMeter

from pproc.common.param_requester import ParamRequester
from pproc.common.utils import delayed_map, dict_product
from pproc.config.base import Parallelisation


class SynchronousExecutor(fut.Executor):
    """Dummy Executor that calls the functions directly"""

    def submit(self, fn, *args, **kwargs):
        f = fut.Future()
        f.set_result(fn(*args, **kwargs))
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

    def __init__(self, n_par: int, queue_size: int = 0, initializer=None, initargs=()):
        """
        :param n_par: number of processes
        :queue_size: maximum number of allowed pending futures, if 0 then
        no queueing is implemented
        """
        super().__init__(max_workers=n_par, initializer=initializer, initargs=initargs)
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


def create_executor(options: Parallelisation) -> fut.Executor:
    return (
        SynchronousExecutor()
        if options.n_par_compute == 1
        else QueueingExecutor(
            options.n_par_compute,
            options.queue_size,
            initializer=signal.signal,
            initargs=(signal.SIGTERM, signal.SIG_DFL),
        )
    )


def parallel_processing(
    process,
    plan,
    n_par,
    initializer=signal.signal,
    initargs=(signal.SIGTERM, signal.SIG_DFL),
):
    """Run a processing function in parallel

    Parameters
    ----------
    process: callable
        Processing function to call. The return value is used as a recovery key.
    plan: iterable of tuples
        Arguments for the processing function
    n_par: int
        Number of parallel processes
    initializer: func
        Function to run before creation of each worker
    initargs: tuple
        Arguments for initializer
    """
    executor = (
        SynchronousExecutor()
        if n_par == 1
        else fut.ProcessPoolExecutor(
            max_workers=n_par, initializer=initializer, initargs=initargs
        )
    )
    with executor:
        for future in fut.as_completed(
            executor.submit(process, *args) for args in plan
        ):
            future.result()


def _retrieve(data_requesters: List[ParamRequester], **kwargs):
    """
    Retrieve data function for multiple data requests
    with retrieve_data method. If requested, grib template messages are written to
    file and their filename returned

    :param data_requesters: list of objects with retrieve_data method
    :param kwargs: keys to retrieve data for (must include step)
    :return: list of retrieved (metadata, data) tuples
    """
    ids = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    with ResourceMeter(f"Retrieve {ids}"):
        collated_data = []
        for requester in data_requesters:
            collated_data.append(requester.retrieve_data(**kwargs))
        return collated_data


def parallel_data_retrieval(
    num_processes: int,
    dims: dict,
    data_requesters: List[ParamRequester],
    initializer=signal.signal,
    initargs=(signal.SIGTERM, signal.SIG_DFL),
):
    """
    Multiprocess retrieve data function from multiple data requests
    with retrieve_data method. If grib_to_file is true then message templates from the requests are
    written to file and the filename returned with the data, else the message template itself is returned.
    If extra_dims is not empty each tuple produced will have the dict of extra keys as its first element.

    :param num_processes: number of processes to use for data retrieval
    :param dims: dimensions to iterate over (must include step)
    :param data_requesters: list of Parameter instances
    :param initializer: function to call on the creation of each worker
    :param initargs: arguments to initializer
    :return: iterator over dims, retrieved data
    """
    executor = (
        SynchronousExecutor()
        if num_processes == 1
        else fut.ProcessPoolExecutor(
            max_workers=num_processes, initializer=initializer, initargs=initargs
        )
    )
    with executor:
        delay = 0 if num_processes == 1 else num_processes
        submit = lambda keys: (
            keys,
            executor.submit(_retrieve, data_requesters, **keys),
        )
        requests = dict_product(dims)
        for keys, future in delayed_map(delay, submit, requests):
            yield keys, future.result()


def sigterm_handler(signum, handler):
    process_id = os.getpid()
    try:
        parent = psutil.Process(process_id)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.terminate()
    sys.exit()
