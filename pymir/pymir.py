import argparse
import cffi
import sys


class Mir:
    _ffi = None
    _lib = None

    _libfile = "libcmir.so"
    _cdefs = """
    struct mir_cfg_t;

    int simple_mir(const char *infile, const char *outfile, struct mir_cfg_t *cfg);

    struct mir_cfg_t *mir_cfg_new(void);
    int mir_cfg_destroy(struct mir_cfg_t *cfg);

    int mir_cfg_set_str(struct mir_cfg_t *cfg, const char *name, const char *val);
    int mir_cfg_set_int(struct mir_cfg_t *cfg, const char *name, int val);
    int mir_cfg_set_long(struct mir_cfg_t *cfg, const char *name, long val);
    int mir_cfg_set_ll(struct mir_cfg_t *cfg, const char *name, long long val);
    int mir_cfg_set_size(struct mir_cfg_t *cfg, const char *name, size_t val);
    int mir_cfg_set_float(struct mir_cfg_t *cfg, const char *name, float val);
    int mir_cfg_set_double(struct mir_cfg_t *cfg, const char *name, double val);

    int mir_cfg_set_int_v(struct mir_cfg_t *cfg, const char *name, int *val, size_t count);
    int mir_cfg_set_long_v(struct mir_cfg_t *cfg, const char *name, long *val, size_t count);
    int mir_cfg_set_ll_v(struct mir_cfg_t *cfg, const char *name, long long *val, size_t count);
    int mir_cfg_set_size_v(struct mir_cfg_t *cfg, const char *name, size_t *val, size_t count);
    int mir_cfg_set_float_v(struct mir_cfg_t *cfg, const char *name, float *val, size_t count);
    int mir_cfg_set_double_v(struct mir_cfg_t *cfg, const char *name, double *val, size_t count);
    """

    @classmethod
    def _load_lib(cls):
        if cls._ffi is not None:
            return
        cls._ffi = cffi.FFI()
        cls._ffi.cdef(cls._cdefs)
        cls._lib = cls._ffi.dlopen(cls._libfile)

    @classmethod
    def simple_mir(cls, infile, outfile, **kwargs):
        cls._load_lib()

        config_ = cls._ffi.gc(cls._lib.mir_cfg_new(), cls._lib.mir_cfg_destroy)
        for k, v in kwargs.items():
            k_ = k.encode("utf-8")
            if isinstance(v, bytes):
                cls._lib.mir_cfg_set_str(config_, k_, cls._ffi.new("char[]", v))
            elif isinstance(v, str):
                cls._lib.mir_cfg_set_str(config_, k_, cls._ffi.new("char[]", v.encode("utf-8")))
            elif isinstance(v, int):
                cls._lib.mir_cfg_set_long(config_, k_, v)
            elif isinstance(v, float):
                cls._lib.mir_cfg_set_long(config_, k_, v)
            elif isinstance(v, list):
                l = len(v)
                assert l > 0
                if isinstance(v[0], int):
                    cls._lib.mir_cfg_set_long_v(config_, k_, cls._ffi.new("long[]", v), l)
                elif isinstance(v[0], float):
                    cls._lib.mir_cfg_set_double_v(config_, k_, cls._ffi.new("double[]", v), l)
                else:
                    raise TypeError(f"unsupported type {type(v[0])!s} for key {k!r}")
            else:
                raise TypeError(f"unsupported type {type(v)!s} for key {k!r}")

        return cls._lib.simple_mir(infile.encode("utf-8"), outfile.encode("utf-8"), config_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", default="input.grib", nargs="?")
    parser.add_argument("outfile", default="output.grib", nargs="?")
    args = parser.parse_args()

    sts = Mir.simple_mir(args.infile, args.outfile, grid="2/2")
    sys.exit(sts)
