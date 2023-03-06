
import argparse
import sys
from typing import Iterable, List, Tuple

import numpy as np

import eccodes

from pproc.common.config import Config, default_parser
from pproc.common.dataset import open_multi_dataset
from pproc.common.io import Target, missing_to_nan, nan_to_missing, target_from_location
from pproc.common.resources import ResourceMeter
from pproc.common.window import Window


def read_ensemble(sources: dict, loc: str, members: int, dtype=np.float32, **kwargs) -> Tuple[eccodes.GRIBMessage, np.ndarray]:
    """Read data from a GRIB file as a single array

    Parameters
    ----------
    sources: dict
        Sources configuration
    loc: str
        Location of the data (file path, named fdb request, ...)
    members: int
        Number of ensemble members to expect
    dtype: numpy data type
        Data type for the result array (default float32)
    kwargs: any
        Extra arguments for backends that support them

    Returns
    -------
    eccodes.GRIBMessage
        GRIB template (first message read)
    numpy array (nfields, npoints)
        Read data
    """
    def set_number(keys):
        if keys.get('type') == 'pf':
            keys['number'] = range(1, members)
    readers = open_multi_dataset(sources, loc, update=set_number, **kwargs)
    template = None
    data = None
    n_read = 0
    for reader in readers:
        with reader:
            message = reader.peek()
            if message is None:
                raise EOFError(f"No data in {loc!r}")
            if template is None:
                template = message
                data = np.empty((members, template.get('numberOfDataPoints')), dtype=dtype)
            for message in reader:
                i = message.get('perturbationNumber', 0)
                data[i, :] = missing_to_nan(message)
                n_read += 1
    if n_read != members:
        raise EOFError(f"Expected {members} fields in {loc!r}, got {n_read}")
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
            tmp = ens[min(j + 1, m - 1), :].copy()
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
    parser.add_argument("--in-ens", required=True, help="Input ensemble source")
    parser.add_argument("--out-quantiles", required=True, help="Output target")
    parser.add_argument("-o", "--out-paramid", required=True)
    return parser


class QuantilesConfig(Config):
    def __init__(self, args: argparse.Namespace, verbose: bool = True):
        super().__init__(args, verbose=verbose)

        self.num_members = self.options.get('num_members', 51)
        self.num_quantiles = self.options.get('num_quantiles', 100)

        self.windows = self.options.get('windows', [{'range': [-1, -1, 0]}])

        self.sources = self.options.get('sources', {})


def main(args: List[str] = sys.argv[1:]):
    parser = get_parser()
    args = parser.parse_args(args)
    config = QuantilesConfig(args)

    res = ResourceMeter()
    for win_params in config.windows:
        window = Window(win_params)
        for step in window.steps:
            label = "" if step == -1 else f"Step {step}: "
            kwargs = {} if step == -1 else {"step": step}
            print(f"{label}Startup: {res.update()!s}")
            template, ens = read_ensemble(config.sources, args.in_ens, config.num_members, **kwargs)
            print(f"{label}Read ensemble: {res.update()!s}")
            target = target_from_location(args.out_quantiles)
            do_quantiles(ens, template, target, args.out_paramid, n=config.num_quantiles)
            print(f"{label}Quantiles: {res.update()!s}")
            del ens


if __name__ == '__main__':
    sys.exit(main())
