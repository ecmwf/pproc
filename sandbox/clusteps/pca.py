
import numpy as np


def gen_steps(start, end, delta, monthly=False):
    """Generate the list of steps

    Parameters
    ----------
    start: int
        First step
    end: int
        Last step
    delta: int
        Interval between steps
    monthly: bool
        If True, delta is scaled by 7 (monthly forecast)

    Returns
    -------
    list[int]
        Step numbers
    """
    delta_eff = delta if not monthly else (7 * delta)
    return list(range(start, end + 1, delta_eff))


def normalise_angles(angles, positive=True):
    """Normalise angles in degrees

    Parameters
    ----------
    angles: numpy array
        Input angles in degrees
    positive: bool
        If True, the return value is in [0, 360), in [-180, 180) otherwise

    Returns
    -------
    numpy array
        Angles in the appropriate range, see ``positive``
    """
    angles = np.asarray(angles) % 360
    if positive:
        return angles
    return angles - 180


def region_mask(lat_n, lat_s, lon_w, lon_e, lat, lon):
    """Mask off a region

    Parameters
    ----------
    lat_n: float
        Latitude of the northern border ([-90, 90])
    lat_s: float
        Latitude of the southern border ([-90, 90])
    lon_w: float
        Longitude of the western border ([0, 360])
    lon_e: float
        Longitude of the eastern border
    lat: numpy array (npoints)
        Latitudes ([-90, 90])
    lon: numpy array (npoints)
        Longitudes ([0, 360])

    Returns
    -------
    numpy array (npoints)
        Boolean mask
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    assert lat.shape == lon.shape

    mask = (lat >= lat_s) & (lat <= lat_n)
    if lon_w <= lon_e:
        # Contiguous case
        mask &= (lon >= lon_w) & (lon <= lon_e)
    else:
        # Wrap around
        mask &= (lon >= lon_w) | (lon <= lon_e)

    return mask


def region_weights(lat_n, lat_s, lon_w, lon_e, lat, lon, weights=None):
    """Restrict weights to a region

    Parameters
    ----------
    lat_n: float
        Latitude of the northern border ([-90, 90])
    lat_s: float
        Latitude of the southern border ([-90, 90])
    lon_w: float
        Longitude of the western border ([0, 360])
    lon_e: float
        Longitude of the eastern border
    lat: numpy array (npoints)
        Latitudes ([-90, 90])
    lon: numpy array (npoints)
        Longitudes ([0, 360])
    weights: numpy array (npoints), optional
        Initial weights

    Returns
    -------
    numpy array (npoints)
        Masked weights
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    assert lat.shape == lon.shape

    mask = region_mask(lat_n, lat_s, lon_w, lon_e, lat, lon)

    if weights is None:
        weights = np.ones_like(lat)
    else:
        weights = np.asarray(weights)
        assert weights.shape == lat.shape
    weights[~mask] = 0
    return weights


def lat_weights(lat, weights=None):
    """Compute latitude-dependent weights

    Parameters
    ----------
    lat: numpy array (npoints)
        Latitudes
    weights: numpy array (npoints) or None
        Initial weight

    Returns
    -------
    numpy array (npoints)
        Latitude-corrected weights
    """
    if weights is None:
        weights = np.cos(np.radians(lat))
    else:
        weights = np.asarray(weights)
        assert weights.shape == lat.shape
        weights = weights * np.cos(np.radians(lat))
    weights /= weights.sum()
    return weights


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
        Principal components
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


if __name__ == '__main__':
    import argparse
    import eccodeshl

    parser = argparse.ArgumentParser(description="PCA for ensemble data")
    parser.add_argument('-N', '--num-members', type=int, default=51, help="Number of ensemble members")
    parser.add_argument('-S', '--steps', nargs=3, type=int, required=True, help="Steps (start, stop, delta)")
    parser.add_argument('-b', '--bbox', nargs=4, type=float, default=None, help="Bounding box (N, S, W, E)")
    parser.add_argument('-c', '--clip', default=None, type=float, help="Clip anomalies to this absolute value")
    parser.add_argument('-f', '--factor', default=None, type=float, help="Apply this conversion factor to the input fields")
    parser.add_argument('-m', '--mask', default=None, help="Mask file")
    parser.add_argument('-s', '--spread', required=True, help="Ensemble spread (GRIB)")
    parser.add_argument('num_components', type=int, help="Number of components to extract")
    parser.add_argument('ensemble', help="Ensemble data (GRIB)")
    parser.add_argument('output', help="Output file (NPZ)")
    args = parser.parse_args()

    # Read mask
    mask = None
    if args.mask is not None:
        # format?
        raise NotImplementedError()

    # Read ensemble
    nexp = args.num_members
    monthly = (args.steps[1] - args.steps[0] > 120) or (args.steps[1] == args.steps[0])
    steps = gen_steps(args.steps[0], args.steps[1], args.steps[2], monthly=monthly)
    inv_steps = {s: i for i, s in enumerate(steps)}
    nstep = len(steps)
    with eccodeshl.FileReader(args.ensemble) as reader:
        message = reader.peek()
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

    # Mask off region
    if args.bbox is not None:
        lat_n, lat_s, lon_w, lon_e = normalise_angles(args.bbox)
        mask = region_weights(lat_n, lat_s, lon_w, lon_e, lat, lon, mask)

    # Weight by latitude
    weights = lat_weights(lat, mask)

    # Read ensemble stddev, compute mean spread
    with eccodeshl.FileReader(args.spread) as reader:
        message = next(reader)
        # TODO: check param and level
        ens_spread = mean_spread(message.get_array('values'), weights=weights)

    # Normalise ensemble fields
    if args.factor is not None:
        ens *= args.factor

    # Compute ensemble mean
    ens_mean = ensemble_mean(ens)

    # Compute ensemble anomalies
    ens_anom = ensemble_anomalies(ens, ens_mean=ens_mean, clip=args.clip)
    del ens

    # Compute EOF
    ncomp = args.num_components
    eof, pc, var, tot_var = ensemble_pca(ens_anom, ncomp, weights)

    # Compute principal component info
    nfld = nexp
    eof_sd = np.sqrt(var / nfld)
    var_pct = 100. * var / tot_var
    var_cum = np.cumsum(var_pct)

    data = {
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
    np.savez_compressed(args.output, **data)