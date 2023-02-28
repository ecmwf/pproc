
import argparse
import sys
from typing import Iterable, List, Tuple

import numpy as np

import eccodes

from pproc.common.config import default_parser
from pproc.common.io import Target, missing_to_nan, nan_to_missing, target_factory
from pproc.common.resources import ResourceMeter


def read_ensemble(source: str, members: int, dtype=np.float32) -> Tuple[eccodes.GRIBMessage, np.ndarray]:
    """Read data from a GRIB file as a single array

    Parameters
    ----------
    source: str
        Path to the GRIB file
    members: int
        Number of ensemble members to expect
    dtype: numpy data type
        Data type for the result array (default float32)

    Returns
    -------
    eccodes.GRIBMessage
        GRIB template (first message read)
    numpy array (nfields, npoints)
        Read data
    """
    with eccodes.FileReader(source) as reader:
        template = reader.peek()
        if template is None:
            raise EOFError(f"No data in {source!r}")
        data = np.empty((members, template.get('numberOfDataPoints')), dtype=dtype)
        n_read = 0
        for i, message in enumerate(reader):
            data[i, :] = missing_to_nan(message)
            n_read += 1
    if n_read != members:
        raise EOFError(f"Expected {members} fields in {source!r}, got {n_read}")
    return template, data


def quantiles(ens: np.ndarray, n: int = 100, method: str = 'sort') -> Iterable[np.ndarray]:
    """Compute quantiles

    Parameters
    ----------
    ens: numpy array (nens, npoints)
        Ensemble data
    n: int
        Number of quantiles (default 100 = percentiles)
    method: 'sort', 'numpy_bulk', 'numpy'
        Method of computing the quantiles:
        * sort: sort `ens` in place, then interpolates the quantiles one by one
        * numpy_bulk: compute all the quantiles at once using `numpy.quantile`
        * numpy: compute the quantiles one by one

    Returns
    -------
    Iterable[numpy array]
        Quantiles, in increasing order
    """
    if method not in ('sort', 'numpy_bulk', 'numpy'):
        raise ValueError(f"Invalid method {method!r}, expected 'sort', 'numpy_bulk', or 'numpy'")

    if method == 'numpy_bulk':
        q = np.linspace(0., 1., n + 1)
        quantiles = np.quantile(ens, q, axis=0)
        yield from quantiles
        return

    if method == 'sort':
        ens.sort(axis=0)

    for i in range(n + 1):
        q = i / n
        if method == 'numpy':
            yield np.quantile(ens, q, axis=0)

        elif method == 'sort':
            q = i / n
            m = ens.shape[0]
            f = (m - 1) * q
            j = int(f)
            x = f - j
            quantile = ens[j, :].copy()
            quantile *= 1 - x
            tmp = ens[min(j + 1, m - 1), :]
            tmp *= x
            quantile += tmp
            yield quantile


def do_quantiles(ens: np.ndarray, template: eccodes.GRIBMessage, target: Target, out_paramid: str, n: int = 100):
    """Compute quantiles

    Parameters
    ----------
    ens: numpy array (nens, npoints)
        Ensemble data
    template: eccodes.GRIBMessage
        GRIB template for output
    target: Target
        Target to write to
    n: int
        Number of quantiles (default 100 = percentiles)
    """
    for i, quantile in enumerate(quantiles(ens, n, method='sort')):
        message = template.copy()
        message.set("type", "pb")
        message.set("numberOfForecastsInEnsemble", n)
        message.set("perturbationNumber", i)
        message.set("paramId", out_paramid)
        message.set_array("values", nan_to_missing(message, quantile))
        target.write(message)


def get_parser() -> argparse.ArgumentParser:
    description = "Compute quantiles of an ensemble"
    parser = default_parser(description=description)
    parser.add_argument("--infile", required=True, help="Input ensemble (GRIB)")
    parser.add_argument("--outfile", required=True, help="Output file (GRIB)")
    parser.add_argument("-m", "--members", type=int, default=51)
    parser.add_argument("-n", "--number", type=int, default=100, help="Number of quantiles (default 100 = percentiles)")
    parser.add_argument("-o", "--out-paramid", required=True)
    return parser


def main(args: List[str] = sys.argv[1:]):
    parser = get_parser()
    args = parser.parse_args(args)

    res = ResourceMeter()
    print(f"Startup: {res!s}")
    template, ens = read_ensemble(args.infile, args.members)
    print(f"Read ensemble: {res.update()!s}")
    target = target_factory("file", out_file=args.outfile)
    do_quantiles(ens, template, target, args.out_paramid, n=args.number)
    print(f"Quantiles: {res.update()!s}")


if __name__ == '__main__':
    sys.exit(main())
