
import argparse
from typing import List, Optional
import sys

import numpy as np
import eccodes

from pproc.common import default_parser
from pproc.clustereps.config import ClusterConfigBase
from pproc.clustereps.utils import normalise_angles, lat_weights, region_weights


def read_ensemble_grib(path, steps, nexp):
    """Read ensemble data from a GRIB file

    Parameters
    ----------
    path: path-like
        Path to the GRIB file
    steps: list[int]
        List of steps
    nexp: int
        Number of ensemble members

    Returns
    -------
    numpy array (npoints)
        Latitudes in deg
    numpy array (npoints)
        Longitudes in [0, 360) deg
    numpy array (nexp, nstep, npoints)
        Ensemble data
    eccodes.Message
        Template message
    """
    inv_steps = {s: i for i, s in enumerate(steps)}
    nstep = len(steps)
    template = None
    with eccodes.FileReader(path) as reader:
        message = reader.peek()
        template = message
        npoints = message.get('numberOfDataPoints')
        lat = message.get_array('latitudes')
        lon = normalise_angles(message.get_array('longitudes'))
        ens = np.empty((nexp, nstep, npoints))
        for message in reader:
            iexp = message.get('perturbationNumber')
            step = message.get('step')
            # TODO: check param and level
            istep = inv_steps.get(step, None)
            if istep is not None:
                ens[iexp, istep, :] = message.get_array('values')
    return lat, lon, ens, template


def mean_spread(stddev, weights=None):
    """Compute the mean ensemble spread

    If the weights are provided, this functions assume that their sum is equal
    to 1.

    Parameters
    ----------
    stddev: numpy array (npoints)
        Ensemble standard deviation
    weights: numpy array (npoints) or None
        Weights (see `lat_weights`)

    Returns
    -------
    float
        Mean ensemble spread
    """
    if weights is None:
        weights = 1. / stddev.size
    else:
        weights = np.asarray(weights)
        assert weights.shape == stddev.shape
    return np.sum(stddev**2 * weights)


def ensemble_mean(ens):
    """Compute the ensemble mean

    Parameters
    ----------
    ens: numpy array (..., nexp, nstep, npoints)
        Ensemble fields

    Returns
    -------
    numpy array (..., nstep, npoints)
        Ensemble mean
    """
    assert ens.ndim >= 3
    return ens.mean(axis=-3)


def ensemble_anomalies(ens, ens_mean=None, clip=None):
    """Compute the ensemble anomalies

    Parameters
    ----------
    ens: numpy array (..., nexp, nstep, npoints)
        Ensemble fields
    ens_mean: numpy array (..., nstep, npoints) or None
        Ensemble mean (see `ensemble_mean`), computed if not provided
    clip: float or None
        If specified, clip anomalies to [-clip, clip]

    Returns
    -------
    numpy array (..., nexp, nstep, npoints)
        Ensemble anomalies
    """
    if ens_mean is None:
        ens_mean = ensemble_mean(ens)
    else:
        assert ens_mean.shape[:-2] == ens.shape[:-3]
        assert ens_mean.shape[-2:] == ens.shape[-2:]
    anom = ens - ens_mean[..., np.newaxis, :, :]
    if clip is None:
        return anom
    return np.clip(anom, -clip, clip)


def ensemble_pca(ens_anom, ncomp, weights=None):
    """Perform a principal component analysis on ensemble data

    Parameters
    ----------
    ens_anom: numpy array (..., nexp, nstep, npoints)
        Ensemble anomalies (see `ensemble_anomalies`)
    ncomp: int
        Number of principal components to keep
    weights: numpy array (npoints)
        Weight the grid points by this factor

    Returns
    -------
    numpy array (ncomp, nstep, npoints)
        Empirical Orthogonal Functions computed from the PCA
    numpy array (ncomp, ..., nexp)
        Ensemble anomalies in PC space
    numpy array (ncomp)
        Variance associated with each component (descending order)
    float
        Total variance of the components before truncating
    """
    orig_sh = ens_anom.shape[:-2]
    nstep, npoints = ens_anom.shape[-2:]
    ens = ens_anom.reshape(-1, nstep, npoints)

    if weights is None:
        ens_cov = np.tensordot(ens, ens, axes=((-1, -2), (-1, -2)))
    else:
        ens_cov = np.einsum('l,ikl,jkl->ij', weights, ens, ens)

    if nstep > 1:
        ens_cov /= nstep

    evals, evecs = np.linalg.eigh(ens_cov)
    nfld = evals.shape[0]

    comp_ev = evals[-ncomp:][::-1]
    sum_ev = evals.sum()

    pcens = np.empty((ncomp, nfld))
    for i in range(ncomp):
        pcens[i, :] = evecs[:, -i-1]
    pcens *= np.sqrt(nfld)

    eof = np.tensordot(pcens, ens, axes=1)
    eof /= nfld

    return eof, pcens.reshape((ncomp,) + orig_sh), comp_ev, sum_ev


class PCAConfig(ClusterConfigBase):
    def __init__(self, args):
        super().__init__(args)

        # Normalisation factor (1)
        self.factor = self.options.get('pca_factor', None)
        # Number of components to extract
        self.ncomp = self.options['num_components']


def do_pca(config: PCAConfig, lat: np.ndarray, lon: np.ndarray, ens: np.ndarray, spread: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """Run the ensemble PCA

    Parameters
    ----------
    config: PCAConfig
        PCA configuration
    lat: numpy array (npoints)
        Latitudes
    lon: numpy array (npoints)
        Longitudes
    ens: numpy array (..., nexp, nstep, npoints)
        Ensemble data
    spread: numpy array (npoints)
        Ensemble spread
    mask: numpy array (npoints), optional
        Initial mask

    Returns
    -------
    dict
        PCA data
        * lat: numpy array (npoints)
            Latitudes
        * lon: numpy array (npoints)
            Longitudes
        * mask: numpy array (npoints)
            Applied mask
        * ens_mean: numpy array (..., nstep, npoints)
            Ensemble mean
        * ens_anom: numpy array (..., nexp, nstep, npoints)
            Ensemble in anomaly space
        * eof: numpy array (ncomp, nstep, npoints)
            Empirical orthogonal functions
        * pc: numpy array (ncomp, ..., nexp)
            Principal components of the ensemble
        * eof_sd: numpy array (ncomp)
            EOF standard deviation
        * var_pct: numpy array (ncomp)
            Percentage of variance explained by PCs
        * var_cum: numpy array (ncomp)
            Cumulative percentage of variance explained by PCs
        * ens_spread: float
            Mean ensemble spread
        * weights: numpy array (npoints)
            Applied weights
    """
    # Mask off region
    if config.bbox is not None:
        lat_n, lat_s, lon_w, lon_e = normalise_angles(config.bbox)
        mask = region_weights(lat_n, lat_s, lon_w, lon_e, lat, lon, mask)

    # Weight by latitude
    weights = lat_weights(lat, mask)

    # Compute mean spread
    ens_spread = mean_spread(spread, weights=weights)

    # Normalise ensemble fields
    if config.factor is not None:
        ens *= config.factor

    # Compute ensemble mean
    ens_mean = ensemble_mean(ens)

    # Compute ensemble anomalies
    ens_anom = ensemble_anomalies(ens, ens_mean=ens_mean, clip=config.clip)
    del ens

    # Compute EOF
    eof, pc, var, tot_var = ensemble_pca(ens_anom, config.ncomp, weights)

    # Compute principal component info
    nfld = np.prod(pc.shape[1:])
    eof_sd = np.sqrt(var / nfld)
    var_pct = 100. * var / tot_var
    var_cum = np.cumsum(var_pct)

    return {
        'lat': lat,
        'lon': lon,
        'mask': mask,              # EOF
        'ens_mean': ens_mean,      # EM, per ensemble then step
        'ens_anom': ens_anom,      # AN, per ensemble then member then step
        'eof': eof,                # EOF, per component then step
        'pc': pc,                  # PC, per ensemble then member then component
        'eof_sd': eof_sd,          # SD
        'var_pct': var_pct,        # SD
        'var_cum': var_cum,        # SD
        'ens_spread': ens_spread,  # SD
        'weights': weights,        # EOF
    }


def get_parser() -> argparse.ArgumentParser:
    """initialize command line application argument parser.

    Returns
    -------
    argparse.ArgumentParser
        
    """

    _description='PCA for ensemble data'
    parser = default_parser(description=_description)

    group = parser.add_argument_group('Principal components analysis arguments')

    group.add_argument('-m', '--mask', default=None, help="Mask file")
    group.add_argument('--spread', required=True, help="Ensemble spread (GRIB)")
    group.add_argument('-e', '--ensemble', required=True, help="Ensemble data (GRIB)")
    group.add_argument('-o', '--output', required=True, help="Output file (NPZ)")
   
    return parser


def main(cmdArgs=sys.argv[1:]):

    parser = get_parser()
    args = parser.parse_args(cmdArgs)
    config = PCAConfig(args)

    # Read mask
    if args.mask is not None:
        # format?
        raise NotImplementedError()

    # Read ensemble
    nexp = config.num_members
    lat, lon, ens, template = read_ensemble_grib(args.ensemble, config.steps, nexp)

    # Read ensemble stddev
    with eccodes.FileReader(args.spread) as reader:
        message = next(reader)
        # TODO: check param and level
        spread = message.get_array('values')

    data = do_pca(config, lat, lon, ens, spread, args.mask)

    np.savez_compressed(args.output, **data)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))