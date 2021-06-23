
import numpy as np

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
    #for icl in range(nclim):
    #    for i in range(npoints):
    #        for ifo in range(nens):
    #            if ens[ifo, i] <= clim[icl, i]:
    #                frac[icl, i] += 1
    ##################################
    for icl in range(nclim):
        for ifo in range(nens):
            mask = ens[ifo, :] <= clim[icl, :]
            frac[icl, mask] += 1
    frac /= nens

    # Compute formula coefficients
    p = np.linspace(0., 1., nclim)
    dp = 1 / (nclim - 1)  #np.diff(p)

    acosdiff = np.diff(np.acos(np.sqrt(p)))
    proddiff = np.diff(np.sqrt(p * (1. - p)))

    acoef = (1. - 2. * p[:-1]) * acosdiff + proddiff

    # TODO: handle epsilon
    efi = np.zeros(npoints)
    for icl in range(nclim-1):
        dFdp = (frac[icl+1, :] - frac[icl, :]) / dp
        dEFI = (2. * frac[icl, :] - 1.) * acosdiff[icl] + acoef[icl] * dFdp - proddiff[icl]  # XXX: why proddiff here?!
        efi += dEFI
    efi *= 2. / np.pi

    return efi

