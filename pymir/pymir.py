import argparse
import cffi
import sys


class Mir:
    _ffi = None
    _lib = None

    _libfile = "libcmir.so"
    _cdefs = """
    int simple_mir(const char *infile, const char *outfile);
    """

    @classmethod
    def _load_lib(cls):
        if cls._ffi is not None:
            return
        cls._ffi = cffi.FFI()
        cls._ffi.cdef(cls._cdefs)
        cls._lib = cls._ffi.dlopen(cls._libfile)

    @classmethod
    def simple_mir(cls, infile, outfile):
        cls._load_lib()
        infile_ = cls._ffi.new("char[]", bytes(infile, "utf-8"))
        outfile_ = cls._ffi.new("char[]", bytes(outfile, "utf-8"))
        return cls._lib.simple_mir(infile_, outfile_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", default="input.grib", nargs="?")
    parser.add_argument("outfile", default="output.grib", nargs="?")
    args = parser.parse_args()

    sts = Mir.simple_mir(args.infile, args.outfile)
    sys.exit(sts)
