
from contextlib import ExitStack
import copy
from datetime import date, datetime
import errno
from math import ceil
import os
import select
import shlex
import subprocess
from tempfile import NamedTemporaryFile, mktemp
from typing import List, Union, AnyStr


def _mkftemp(dir=None, mode=0o600, num_attempts=1000):
    for _ in range(num_attempts):
        path = mktemp(dir=dir)
        try:
            os.mkfifo(path, mode)
        except OSError as e:
            if e.errno == errno.EEXIST:
                continue
            raise
        return path
    raise OSError(errno.EEXIST, f"_mkftemp: No usable temporary name found after {num_attempts} attempts")


class FIFO:
    def __init__(self, path=None, mode=0o600, flags=os.O_RDONLY, dir=None, delete=True):
        self.delete = delete
        self.closed = False
        if path is not None:
            os.mkfifo(path, mode)
        else:
            path = _mkftemp(dir=dir, mode=mode)
        self.path = path
        self.fd = os.open(path, flags | os.O_NONBLOCK)
        # Don't wait for the other end of the pipe to be opened, but block on read
        os.set_blocking(self.fd, True)

    def fileno(self):
        if self.closed:
            raise IOError("Operation on a closed FIFO")
        return self.fd

    def wait(self, timeout=None):
        if self.closed:
            raise IOError("Operation on a closed FIFO")
        return len(select.select([self.fd], [], [], timeout)[0]) > 0

    def ready(self):
        return self.wait(0)

    def read(self, size):
        if self.closed:
            raise IOError("Operation on a closed FIFO")
        return os.read(self.fd, size)

    def close(self):
        if self.closed:
            return
        os.close(self.fd)
        self.closed = True
        if self.delete:
            os.unlink(self.path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()


class SubProcess(subprocess.Popen):
    def check(self):
        retcode = self.poll()
        if retcode is None:
            return True
        if retcode == 0:
            return False
        raise subprocess.CalledProcessError(retcode, self.args)

    def running(self):
        return self.poll() is None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.wait(1)
        except subprocess.TimeoutExpired:
            self.kill()


class MARSReader:
    def __init__(self, proc: SubProcess, fifo: FIFO, stack=None):
        if stack is None:
            stack = ExitStack()
            stack.push(proc)
            stack.push(fifo)
        self.stack = stack
        self.proc = proc
        self.fifo = fifo

    def read(self, size: int) -> bytes:
        # Avoid empty reads when the writer has not yet opened the FIFO
        # After that, will block iff 1. the writing end is open 2. there is no data to read
        while self.proc.running():
            if self.fifo.wait(1e-2):
                # We have data (or EOF)
                break
        buf = b''
        # Don't block again if there is nothing to read
        if self.fifo.ready():
            while len(buf) < size:
                chunk = self.fifo.read(size - len(buf))
                if not chunk:
                    break
                buf += chunk
        if not buf:
            self.proc.check()
        return buf

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stack.__exit__(exc_type, exc_value, traceback)

    def close(self):
        self.__exit__(None, None, None)


def _val_to_mars(val):
    if isinstance(val, bytes):
        return val
    elif isinstance(val, (list, tuple)):
        return b"/".join(_val_to_mars(v) for v in val)
    elif isinstance(val, str):
        pass
    elif isinstance(val, int):
        val = str(val)
    elif isinstance(val, (datetime, date)):
        val = val.strftime("%Y%m%d")
    elif isinstance(val, range):
        first = val.start
        last = val.start + (ceil((val.stop - val.start) / val.step) - 1) * val.step
        step = val.step
        val = f"{first}/to/{last}"
        if step != 1:
            val += f"/by/{step}"
    else:
        raise TypeError(f"Cannot convert {type(val)} to MARS request")
    return val.encode("utf-8")


def to_mars(verb: str, req: dict, return_bytes: bool = False) -> AnyStr:
    def _gen_req():
        yield verb.encode("utf-8")
        for key, val in req.items():
            key = key.encode("utf-8")
            val = _val_to_mars(val)
            yield key + b"=" + val
    ret = b",".join(_gen_req())
    return ret if return_bytes else ret.decode("utf-8")


def mars_retrieve(req: dict, mars_cmd: Union[str, List[str]] = "mars", tmpdir=None) -> MARSReader:
    req = copy.deepcopy(req)
    with ExitStack() as stack:
        req_file = stack.enter_context(NamedTemporaryFile(dir=tmpdir))
        fifo = stack.enter_context(FIFO(dir=tmpdir))
        req['target'] = '"' + fifo.path + '"'
        req_s = to_mars("retrieve", req, return_bytes=True)
        req_file.write(req_s + b"\n")
        req_file.flush()
        cmd = shlex.split(mars_cmd) if isinstance(mars_cmd, str) else mars_cmd
        cmd.append(req_file.name)
        proc = stack.enter_context(SubProcess(cmd, stdin=subprocess.DEVNULL))
        return MARSReader(proc, fifo, stack=stack.pop_all())