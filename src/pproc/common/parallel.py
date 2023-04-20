
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

