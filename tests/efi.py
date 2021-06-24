import numba
import numpy as np
from numba import float64, float32

@numba.jit(float64[:](float64[:,:], float64[:,:]), fastmath=False, nopython=True, nogil=True, cache=True)
def efi(clim, ens):
    """Compute EFI

    Parameters
    ----------
    clim: numpy array (nclim, npoints)
        Sorted per-point climatology
    ens: numpy array (nens, npoints)
        Ensemble forecast

    Returns
    -------
    numpy array (npoints)
        EFI values
    """

    # Compute fraction of the forecast below climatology
    nclim, npoints = clim.shape
    nens, npoints_ens = ens.shape
    assert npoints == npoints_ens
    frac = np.zeros_like(clim)
    ##################################
    for ifo in numba.prange(nens):
        for icl in range(nclim):
            for i in range(npoints):
               if ens[ifo, i] <= clim[icl, i]:
                   frac[icl, i] += 1
    ##################################
    # for icl in range(nclim):
    #     for ifo in range(nens):
    #         mask = ens[ifo, :] <= clim[icl, :]
    #         frac[icl, mask] += 1
    ##################################
    # for icl in range(nclim):
    #     frac[icl, :] = np.sum(ens[:, :] <= clim[icl, np.newaxis, :], axis=0)
    ##################################
    frac /= nens

    # Compute formula coefficients
    p = np.linspace(0., 1., nclim)
    dp = 1 / (nclim - 1)  #np.diff(p)

    acosdiff = np.diff(np.arccos(np.sqrt(p)))
    proddiff = np.diff(np.sqrt(p * (1. - p)))

    acoef = (1. - 2. * p[:-1]) * acosdiff + proddiff

    # TODO: handle epsilon
    efi = np.zeros(npoints)
    ##################################
    # for icl in range(nclim-1):
    #     dFdp = (frac[icl+1, :] - frac[icl, :]) / dp
    #     dEFI = (2. * frac[icl, :] - 1.) * acosdiff[icl] + acoef[icl] * dFdp - proddiff[icl]  # XXX: why proddiff here?!
    #     efi += dEFI
    # efi *= 2. / np.pi
    ##################################
    for icl in numba.prange(nclim-1):
        for i in range(npoints):
            dFdp = (frac[icl+1, i] - frac[icl, i]) / dp
            dEFI = (2. * frac[icl, i] - 1.) * acosdiff[icl] + acoef[icl] * dFdp - proddiff[icl]  # XXX: why proddiff here?!
            efi[i] += dEFI
    efi *= 2. / np.pi

    return efi
