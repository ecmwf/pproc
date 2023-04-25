import concurrent.futures as fut


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
            for future in self.futures:
                if future.done():
                    future.result()
                    self.futures.remove(future)

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
