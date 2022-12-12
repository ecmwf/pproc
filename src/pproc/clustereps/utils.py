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
