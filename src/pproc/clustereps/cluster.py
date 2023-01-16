
import argparse
import concurrent.futures as fut
import functools
import sys

import numpy as np
import numpy.random as npr

from eccodes import FileReader, GRIBMessage

from pproc.common import default_parser
from pproc.clustereps.config import ClusterConfigBase
from pproc.clustereps.utils import lat_weights, region_weights


def read_steps_grib(path, steps):
    """Read multi-step data from a GRIB file

    Parameters
    ----------
    path: path-like
        Path to the GRIB file
    steps: list[int]
        List of steps

    Returns
    -------
    numpy array (nstep, npoints)
        Read data
    """
    inv_steps = {s: i for i, s in enumerate(steps)}
    nstep = len(steps)
    with FileReader(path) as reader:
        message = reader.peek()
        npoints = message.get('numberOfDataPoints')
        data = np.empty((nstep, npoints))
        for message in reader:
            step = message.get('step')
            # TODO: check param and level
            istep = inv_steps.get(step, None)
            if istep is not None:
                data[istep, :] = message.get_array('values')
    return data


def disc_stat(xs, ndis):
    """Compute statistics from a data series divided into discontinuous
    sub-series

    Parameters
    ----------
    xs: numpy array (nt)
        Data series
    ndis: int
        Number of sub-series

    Returns
    -------
    float
        Average
    float
        Standard deviation
    float
        Auto-correlation
    """
    xs = np.asarray(xs)
    assert xs.ndim == 1
    nx = xs.shape[0]

    avg = np.mean(xs)

    ns = nx // max(1, ndis)
    var = np.var(xs[:ns * ndis])
    xs = xs - avg
    cov = 0.
    for i in range(ndis):
        js = i * ns
        je = js + ns - 1
        cov += np.sum(xs[js:je] * xs[js+1:je+1])

    sd = np.sqrt(var)
    ac = cov / (var * (nx - ndis))
    return avg, sd, ac


def prepare_data(data: dict, npc: int, factor: float = 1., verbose: bool = False):
    """Prepare PCA data

    Parameters
    ----------
    data: dict
        PCA data, see `pproc.clustereps.pca`
    npc: int
        Number of principal components to use
    factor: float, optional
        Normalisation factor for the ensemble spread
    verbose: bool (default False)
        If True, print diagnostics

    Returns
    -------
    numpy array (npc, nfld)
        Principal components of the ensemble, shifted to zero mean
    numpy array (neof)
        EOF standard deviation
    float
        Ensemble spread, scaled
    numpy array (nfld, nsteps, npoints)
        Ensemble data in anomaly space
    numpy array (nsteps, npoints)
        Ensemble mean
    numpy array (npc)
        Standard deviation of the PCs
    numpy array (npc)
        Auto-correlation of the PCs
    """
    pc = data['pc'][:npc, ...].reshape((npc, -1))
    eof_sd = data['eof_sd']

    pc *= eof_sd[:npc, np.newaxis]

    ens_spread = data['ens_spread']
    if verbose:
        print(f"Ensemble spread: {ens_spread}")
    ens_spread *= factor
    if verbose:
        print(f"Ensemble spread after rescaling: {ens_spread}")

    nstep, ngp = data['ens_anom'].shape[-2:]
    ens_anom = data['ens_anom'].reshape((-1, nstep, ngp))
    ens_mean = data['ens_mean'].reshape((nstep, ngp))

    # Compute PC statistics and subtract mean
    ndis = 1
    pc_mean = np.empty(npc)
    pc_sd = np.empty(npc)
    pc_ac = np.empty(npc)
    for i in range(npc):
        pc_mean[i], pc_sd[i], pc_ac[i] = disc_stat(pc[i, :], ndis)
    pc -= pc_mean[:, np.newaxis]

    return pc, eof_sd, ens_spread, ens_anom, ens_mean, pc_sd, pc_ac


class Random:
    """Pseudo-random number generator

    Parameters
    ----------
    seed: int
        Initial state for the generator

    References
    ----------
    Numerical Recipes, Chapter 7.1
    """

    M = 714025
    A = 1366
    C = 150889

    def __init__(self, seed):
        self.current = -1
        self.shuffle = np.empty(97, dtype=np.int32)

        seed = int(seed)
        self.state = (self.C + abs(seed)) % self.M

        for i in range(97):
            self.state = (self.A * self.state + self.C) % self.M
            self.shuffle[i] = self.state

        self.state = (self.A * self.state + self.C) % self.M
        self.current = self.state

    def random(self):
        """Generate a uniform number in [0, 1)"""

        # Get one integer number from the shuffle table
        i = (97 * self.current) // self.M
        self.current = self.shuffle[i]

        # Turn the selected integer into a real no. between 0 and 1
        res = self.current / self.M

        # Replace the selected integer with another random integer
        self.state = (self.A * self.state + self.C) % self.M
        self.shuffle[i] = self.state

        return res

    def rand(self):
        """Generate a uniform number in [0, 1)"""
        return self.random()

    def randrange(self, start_or_stop, stop=None, step=None):
        """Generate an integer in the given range

        See also `random.randrange`"""
        if stop is None:
            start = 1
            stop = int(start_or_stop)
        else:
            start = int(start_or_stop)
            stop = int(stop)
        width = stop - start
        if step is None:
            return start + int(width * self.random())
        else:
            step = int(step)
            num = (width + step - 1) // step
            return start + step * int(num * self.random())

    def randint(self, start_or_stop, stop=None):
        """Generate an integer between the given bounds

        See also `numpy.random.RandomState.randint`"""
        return self.randrange(start_or_stop, stop)


def init_rand(pc: np.ndarray) -> npr.RandomState:
    """Initialise the random state in a data-dependent manner"""
    rand = Random(-11111)
    pc0 = pc[0, pc.shape[1] // 2]
    for x in pc[0, :]:
        if x >= pc0:
            rand.random()
    return npr.RandomState(rand.randrange(1 << 32))


def select_seeds(ncl, pc, r2seed, d2seed, rand):
    """Select clustering seeds

    The seeds must satisfy the following conditions:
    a) have norm less than a maximum value in full PC space;
    b) have a minimum distance from each other in full PC space;
    c) belong to different sectors of the PC1-PC2 plane.

    Parameters
    ----------
    ncl: int
        Number of clusters
    pc: numpy array (npc, nfld)
        Principal components
    r2seed: float
        Max squared norm of seeds
    d2seed: float
        Minimum squared distance between seeds
    rand: `Random` or `numpy.random.RandomState`
        Random number generator

    Returns
    -------
    numpy array (ncl, dtype=int)
        Indexes of the selected seeds in the `pc` array (second axis)
    """
    cospn = np.cos(np.pi / ncl)
    npc, nfld = pc.shape
    assert npc >= 2
    indexes = np.empty(ncl, dtype=int)

    # Select first seed
    norm = r2seed + 1
    while norm > r2seed:
        jseed = rand.randint(nfld)
        norm = np.sum(np.square(pc[:, jseed]))

    indexes[0] = jseed

    # Select subsequent seeds according to the criteria:
    # a) the squared distance between two seeds must be > d2seed,
    # b) the angular distance in the PC1-PC2 plane must be > pi / ncl.
    for jcl in range(1, ncl):
        costmax = 0
        cost1 = 1 / jcl
        for _ in range(10):
            while jseed in indexes[:jcl]:
                jseed = rand.randint(nfld)

            costf = 0.001

            norm = np.sum(np.square(pc[:, jseed]))
            if norm <= r2seed:
                costf += 1.

            r2j = pc[0, jseed]**2 + pc[1, jseed]**2

            for kcl in range(jcl):
                kseed = indexes[kcl]
                d2 = np.sum(np.square(pc[:, jseed] - pc[:, kseed]))
                if d2 > d2seed:
                    costf += cost1

                dotp = pc[0, jseed] * pc[0, kseed] + pc[1, jseed] + pc[1, kseed]
                r2k = pc[0, kseed]**2 + pc[1, kseed]**2
                if dotp < cospn * np.sqrt(r2j * r2k):
                    costf += cost1

            if costf > costmax:
                indexes[jcl] = jseed
                costmax = costf

    return indexes


def compute_partition(pc, indexes, max_iter=100):
    """Compute cluster partition

    This is equivalent to a single run of the K-means algorithm.

    Parameters
    ----------
    pc: numpy array (npc, nfld)
        Principal components
    indexes: numpy array (ncl, dtype=int)
        Indexes of the seeds in `pc` (second axis)
    max_iter: int
        Maximum number of iterations

    Returns
    -------
    numpy array (nfld, dtype=int)
        Cluster index for each field
    numpy array (ncl, dtype=int)
        Number of fields for each cluster
    float
        Internal variance of the partition
    numpy array (ncl, npc)
        Cluster centroids in PC space
    """
    _, nfld = pc.shape
    ncl, = indexes.shape

    ind_cl = np.zeros(nfld, dtype=int)
    n_fields = np.zeros(ncl, dtype=int)

    centroids = pc.T[indexes, :]

    for _ in range(max_iter):
        n_changed = 0

        for jfld in range(nfld):
            jmin = 0
            d2min = np.sum(np.square(pc[:, jfld] - centroids[0, :]))

            for jcl in range(1, ncl):
                d2 = np.sum(np.square(pc[:, jfld] - centroids[jcl, :]))
                if d2 < d2min:
                    jmin = jcl
                    d2min = d2

            if jmin != ind_cl[jfld]:
                n_changed += 1
            ind_cl[jfld] = jmin

        if n_changed == 0:
            break

        centroids[:, :] = 0

        n_fields = np.bincount(ind_cl, minlength=ncl)
        centroids = np.apply_along_axis(
            lambda x: np.bincount(ind_cl, weights=x, minlength=ncl),
            1,
            pc,
        )
        centroids /= n_fields[:, np.newaxis]

    var = np.sum(np.square(pc.T - centroids[ind_cl, :]))

    return ind_cl, n_fields, var, centroids


def compute_partition_skl(pc, indexes, max_iter=100):
    """Compute cluster partition

    Equivalent to `compute_partition` using scikit-learn.

    Parameters
    ----------
    pc: numpy array (npc, nfld)
        Principal components
    indexes: numpy array (ncl, dtype=int)
        Indexes of the seeds in `pc` (second axis)
    max_iter: int
        Maximum number of iterations

    Returns
    -------
    numpy array (nfld, dtype=int)
        Cluster index for each field
    numpy array (ncl, dtype=int)
        Number of fields for each cluster
    float
        Internal variance of the partition
    numpy array (ncl, npc)
        Cluster centroids in PC space
    """
    from sklearn.cluster import k_means

    ncl, = indexes.shape
    init = pc.T[indexes, :]
    centroids, ind_cl, var = k_means(pc.T, ncl, init=init, n_init=1, max_iter=max_iter)

    n_fields = np.bincount(ind_cl, minlength=ncl)

    return ind_cl, n_fields, var, centroids


def full_clustering(ncl, npass, pc, rand, max_iter=100, rseed=1.5, dseed=0.5):
    """Partition the data into clusters

    Several clustering passes (see `compute_partition`) are run with different
    randomly-selected seeds (see `select_seeds`). The optimal partition is the
    one that minimises internal variance.

    Parameters
    ----------
    ncl: int
        Number of clusters
    npass: int
        Number of clustering passes
    pc: numpy array (npc, nfld)
        Principal components
    rand: `Random` or `numpy.random.RandomState`
        Random number generator
    max_iter: int
        Maximum number of iterations of the clustering algorithm
    rseed: float
        Maximum norm of seeds (fraction of total PC variance)
    dseed: float
        Minimum distance between seeds (fraction of total PC variance)

    Returns
    -------
    numpy array (nfld, dtype=int)
        Cluster index for each field
    numpy array (ncl, dtype=int)
        Number of fields for each cluster
    tuple[float, float, float]
        Centroid variance, internal variance and variance ratio of the optimal
        partition
    numpy array (ncl, npc)
        Cluster centroids in PC space
    numpy array (npass, ncl, dtype=int)
        Seed indexes for each pass
    """
    _, nfld = pc.shape
    var_tot = np.sum(np.square(pc)) / nfld
    r2seed = rseed**2 * var_tot
    d2seed = dseed**2 * var_tot

    indexes = np.empty((npass, ncl), dtype=int)

    # Run clustering passes
    i_opt = None
    var_opt = None
    for i in range(npass):
        indexes[i, :] = select_seeds(ncl, pc, r2seed, d2seed, rand)
        _, _, var, _ = compute_partition_skl(pc, indexes[i, :], max_iter)
        #print(f"DBG: Pass {i}, variance: {var}")

        if i == 0 or var < var_opt:
            i_opt = i
            var_opt = var

    # Recompute optimal partition
    ind_cl, n_fields, var, centroids = \
        compute_partition_skl(pc, indexes[i_opt, :], max_iter)
    #print(f"DBG: Final variance: {var}")

    var_cen = np.sum(n_fields * np.sum(np.square(centroids), axis=1))
    var_ratio = var_cen / var

    return ind_cl, n_fields, (var_cen, var, var_ratio), centroids, indexes


def full_clustering_skl(ncl, npass, pc, rand, max_iter=100, rseed=1.5, dseed=0.5):
    """Partition the data into clusters

    Several clustering passes (see `compute_partition`) are run with different
    randomly-selected seeds (see `select_seeds`). The optimal partition is the
    one that minimises internal variance.

    Parameters
    ----------
    ncl: int
        Number of clusters
    npass: int
        Number of clustering passes
    pc: numpy array (npc, nfld)
        Principal components
    rand: `numpy.random.RandomState`
        Random number generator
    max_iter: int
        Maximum number of iterations of the clustering algorithm
    rseed: float
        For compatibility only
    dseed: float
        For compatibility only

    Returns
    -------
    numpy array (nfld, dtype=int)
        Cluster index for each field
    numpy array (ncl, dtype=int)
        Number of fields for each cluster
    tuple[float, float, float]
        Centroid variance, internal variance and variance ratio of the optimal
        partition
    numpy array (ncl, npc)
        Cluster centroids in PC space
    None
        For compatibility only
    """
    from sklearn.cluster import k_means
    _, nfld = pc.shape

    centroids, ind_cl, var = k_means(pc.T, ncl, n_init=npass, max_iter=max_iter, random_state=rand)

    n_fields = np.bincount(ind_cl, minlength=ncl)

    var_cen = np.sum(n_fields * np.sum(np.square(centroids), axis=1))
    var_ratio = var_cen / var

    return ind_cl, n_fields, (var_cen, var, var_ratio), centroids, None


def sort_clusters(n_fields, ind_cl, centroids):
    """Sort clusters in decreasing frequency order

    Parameters
    ----------
    n_fields: numpy array (ncl, dtype=int)
        Number of fields for each cluster
    ind_cl: numpy array(nfld, dtype=int)
        Cluster index for each field
    centroids: numpy array (ncl, npc)
        Cluster centroids in PC space

    Returns
    -------
    numpy array (ncl, dtype=int)
        `n_fields`, sorted
    numpy array(nfld, dtype=int)
        `ind_cl`, updated with new indices
    numpy array (ncl, npc)
        `centroids`, sorted
    """
    indexes = np.argsort(n_fields)[::-1]
    indexes_inv = np.empty_like(indexes)
    for i, j in enumerate(indexes):
        indexes_inv[j] = i
    return n_fields[indexes], indexes_inv[ind_cl], centroids[indexes, :]


def gauss_series(n, avg, sd, ac, ndis, rand):
    """Generate normally-distributed series

    Parameters
    ----------
    n: int
        Number of data points in the series
    avg: float
        Series expected average
    sd: float
        Series expected standard deviation
    ac: float or None
        Series expected auto-correlation
    ndis: int
        Number of discontinuous sub-series
    rand: `Random` or `numpy.random.RandomState`
        Random number generator

    Returns
    -------
    numpy array (n)
        Generated series

    References
    ----------
    Numerical Recipes, Chapter 7.2
    """
    xs = np.empty(n)

    # Generate a series of n gaussian deviates
    for i in range(0, n, 2):
        r2 = 0.
        while r2 > 1. or r2 == 0.:
            u = 2. * rand.rand() - 1.
            v = 2. * rand.rand() - 1.
            r2 = u * u + v * v
        fact = np.sqrt(-2. * np.log(r2) / r2)
        xs[i] = u * fact
        if i < n - 1:
            xs[i+1] = v * fact

    # Introduce autocorrelation if requested
    if ac is not None:
        ns = n // max(1, ndis)
        sd2 = np.sqrt(1. - ac * ac)

        for i in range(ndis):
            js = i * ns
            je = js + ns
            xs[js+1:je] = ac * xs[js:je-1] + sd2 * xs[js+1:je]

    # Set assigned average and standard deviation
    xs *= sd
    xs += avg

    return xs


def gauss_series_np(n, avg, sd, ac, ndis, rand):
    """Generate normally-distributed series

    Parameters
    ----------
    n: int
        Number of data points in the series
    avg: float
        Series expected average
    sd: float
        Series expected standard deviation
    ac: float or None
        Series expected auto-correlation
    ndis: int
        Number of discontinuous sub-series
    rand: `numpy.random.RandomState`
        Random number generator

    Returns
    -------
    numpy array (n)
        Generated series
    """
    xs = rand.normal(size=n)

    # Introduce autocorrelation if requested
    if ac is not None:
        ns = n // max(1, ndis)
        sd2 = np.sqrt(1. - ac * ac)

        for i in range(ndis):
            js = i * ns
            je = js + ns
            xs[js+1:je] = ac * xs[js:je-1] + sd2 * xs[js+1:je]

    # Set assigned average and standard deviation
    xs *= sd
    xs += avg

    return xs


def red_noise_cluster_iteration(ncl_max, npass, npc, nfld, pc_sd, pc_ac, rand):
    """Perform clustering on a red noise sample

    Parameters
    ----------
    ncl_max: int
        Maximum number of clusters
    npass: int
        Number of clustering passes
    npc: int
        Number of generated principal components
    nfld: int
        Dimension of the sample space
    pc_sd: numpy array (npc)
        Expected standard deviation of the PCs
    pc_ac: numpy array (npc)
        Expected autocorrelation of the PCs
    rand: `numpy.random.RandomState`
        Random number generator

    Returns
    -------
    numpy array (ncl_max - 1)
        Variance ratio of the partitions, index is number of clusters - 1
    """
    ndis = 1
    pc_red = np.empty((npc, nfld))
    for j in range(npc):
        ts = gauss_series_np(nfld, 0., pc_sd[j], pc_ac[j], ndis, rand)
        tsm = np.mean(ts)
        pc_red[j, :] = ts - tsm

    noise_var = np.zeros(ncl_max - 1)
    for ncl in range(2, ncl_max + 1):
        _, _, var, _, _ = full_clustering_skl(ncl, npass, pc_red, rand)
        noise_var[ncl-2] = var[2]
    return noise_var


def red_noise_cluster(n_samples, ncl_max, npass, npc, nfld, pc_sd, pc_ac, rand, n_par=1):
    """Perform clustering on red noise samples

    Parameters
    ----------
    n_samples: int
        Number of red noise samples to generate
    ncl_max: int
        Maximum number of clusters
    npass: int
        Number of clustering passes
    npc: int
        Number of generated principal components
    nfld: int
        Dimension of the sample space
    pc_sd: numpy array (npc)
        Expected standard deviation of the PCs
    pc_ac: numpy array (npc)
        Expected autocorrelation of the PCs
    rand: `numpy.random.RandomState`
        Random number generator
    n_par: int
        Number of parallel processes

    Returns
    -------
    numpy array (n_samples, ncl_max - 1)
        Variance ratio of the partitions, second index is number of clusters - 1
    """
    if n_par == 1:
        noise_var = np.zeros((n_samples, ncl_max - 1))
        for i in range(n_samples):
            noise_var[i, :] = red_noise_cluster_iteration(ncl_max, npass, npc, nfld, pc_sd, pc_ac, rand)
        return noise_var
    else:
        sample = functools.partial(red_noise_cluster_iteration, ncl_max, npass, npc, nfld, pc_sd, pc_ac)
        seed = rand.randint((1 << 32) - n_samples)
        with fut.ProcessPoolExecutor(max_workers=n_par) as executor:
            rands = (npr.RandomState(seed + i) for i in range(n_samples))
            noise_var = list(executor.map(sample, rands))
        return np.array(noise_var)


class FileTarget:
    def __init__(self, path, mode="wb"):
        self.file = open(path, mode)

    def __enter__(self):
        return self.file.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.file.__exit__(exc_type, exc_value, traceback)

    def write(self, message):
        message.write_to(self.file)


def write_cluster_grib(steps, ind_cl, rep_members, det_index, data, target, keys):
    """Write clustering data to a GRIB output

    Parameters
    ----------
    steps: iterable
        List of steps as (start, end or None)
    ind_cl: numpy array(nfld, dtype=int)
        Cluster index for each field
    rep_members: numpy array(ncl, dtype=int)
        Representative member for each cluster
    det_index: int
        Index of the cluster containing the deterministic forecast
    data: numpy array(nsteps, ncl, ngp)
        Data to write in grid point space
    target: any object with a ``write(eccodes.Message)`` method
        Write target
    keys: dict-like
        GRIB keys to set
    """
    ncl = len(rep_members)
    sample = GRIBMessage.from_samples("clusters_grib1").copy()
    for key, val in keys.items():
        sample.set(key, val)
    sample.set('totalNumberOfClusters', ncl)
    sample.set('controlForecastCluster', ind_cl[0] + 1)
    for icl in range(ncl):
        members = np.nonzero(ind_cl == icl)[0]

        message = sample.copy()
        message.set('clusterNumber', icl + 1)
        message.set('numberOfForecastsInCluster', len(members))
        message.set_array('ensembleForecastNumbers', members)
        message.set('operationalForecastCluster', det_index + 1)
        message.set('representativeMember', rep_members[icl])

        for i, (start, end) in enumerate(steps):
            if end is None:
                message.set('step', start)
            else:
                message.set('startStep', start)
                message.set('endStep', end)
                message.set('stepRange', f"{start}-{end}")

            message.set_array('values', data[i][icl])

            target.write(message)


class ClusterConfig(ClusterConfigBase):
    def __init__(self, args, verbose=True):

        super().__init__(args, verbose=verbose)

        # Variance threshold
        self.var_th = self.options['var_th']
        # Number of PCs to use, optional
        self.npc = self.options.get('npc')
        # Normalisation factor (2/5)
        self.factor = self.options.get('cluster_factor', 0.4)
        # Max number of clusters
        self.ncl_max = self.options['ncl_max']
        # Number of clustering passes
        self.npass = self.options['npass']
        # Number of red-noise samples for significance computation
        self.nrsamples = self.options['nrsamples']
        # Maximum significance threshold
        self.max_sig = self.options['max_sig']
        # Medium significance threshold
        self.med_sig = self.options['med_sig']
        # Minimum significance threshold
        self.min_sig = self.options['min_sig']
        # Significance tolerance
        self.sig_tol = self.options['sig_tol']
        # Parallel red-noise sampling
        self.n_par = self.options.get('n_par', 1)


def select_npc(var_th, var_cum) -> int:
    """Select the number of PCs according to config"""
    for i, var in enumerate(var_cum):
        if var >= var_th:
            return i + 1
    raise ValueError("Not enough PCs to attain the threshold")


def compute_variance_thresholds(ncl_max: int, npc: int, pc_sd: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Compute the variance thresholds

    Parameters
    ----------
    ncl_max: int
        Maximum number of clusters
    npc: int
        Number of PCs selected
    pc_sd: numpy array (ncomp)
        PC standard deviations
    verbose: bool
        If True, print diagnostics

    Returns
    -------
    numpy array (ncl_max+1)
        Variance thresholds for each number of clusters
    """
    cum_pc_var = np.cumsum(np.square(pc_sd))
    if verbose:
        tot_var_all = cum_pc_var[-1]
        tot_sd = np.sqrt(tot_var_all)
        print(f"Total variance: {tot_var_all}, total spread: {tot_sd}")

    tot_var = cum_pc_var[npc - 1]
    if verbose:
        print(f"Total variance explained by PCs: {tot_var}")

    sig_thr = np.zeros(ncl_max+1)
    sig_thr[2:] = cum_pc_var[:ncl_max-1] / tot_var
    sig_thr[2:] *= 4. * (np.arange(1, ncl_max, dtype=sig_thr.dtype)
                        / np.square(np.arange(2, ncl_max + 1, dtype=sig_thr.dtype)))

    return sig_thr


def compute_clusters(ens_anom: np.ndarray, ens_mean: np.ndarray, pc: np.ndarray, ncl_max: int, npass: int, rand: npr.RandomState, verbose: bool = False):
    nfld, nstep, ngp = ens_anom.shape
    ind_cl = [None, None]  # [ncl]
    n_fields = [None, None]  # [ncl]
    var_opt = [None, None]  # [ncl]
    centroids = [None, None]  # [ncl]
    rep_members = [None, None]  # [ncl][jcl]
    centroids_gp = []  # [step][ncl][jcl]
    rep_members_gp = []  # [step][ncl][jcl]
    for i in range(nstep):
        # TODO: write ensemble means to centroids and representative members files?
        step_centroids_gp = [None, None]  # [ncl][jcl]
        step_rep_members_gp = [None, None]  # [ncl][jcl]
        for ncl in range(2, ncl_max + 1):
            if i == 0:
                cur_ind_cl, cur_n_fields, cur_var_opt, cur_centroids, _ = \
                    full_clustering_skl(ncl, npass, pc, rand)
                cur_var_opt = (cur_var_opt[0] / nfld, cur_var_opt[1] / nfld, cur_var_opt[2])
                cur_n_fields, cur_ind_cl, cur_centroids = \
                    sort_clusters(cur_n_fields, cur_ind_cl, cur_centroids)
                ind_cl.append(cur_ind_cl)
                n_fields.append(cur_n_fields)
                var_opt.append(cur_var_opt)
                centroids.append(cur_centroids)
                if verbose:
                    print(f"Partition: {ncl}, internal variance: {cur_var_opt[1]}")

            part_rep_members = []
            part_centroids_gp = []
            part_rep_members_gp = []
            for jcl in range(ncl):
                rn = 1 / n_fields[ncl][jcl]
                centgp = np.zeros(ngp, dtype=ens_anom.dtype)
                # Compute the representative member from the enesemble mean
                first = True
                rmmin = None
                ifld = None
                for jfld in range(nfld):
                    rms = 0.
                    if ind_cl[ncl][jfld] == jcl:
                        centgp += (ens_anom[jfld, i, :] + ens_mean[i, :]) * rn
                        rms = np.sqrt(np.mean(np.square(pc[:, jfld] - centroids[ncl][jcl, :])))
                        if first or rms < rmmin:
                            first = False
                            rmmin = rms
                            ifld = jfld

                # TODO: write centroid in GP space
                # TODO: write representative member (ens_anom[ifld, i, :] + ens_mean[i, :])

                part_centroids_gp.append(centgp)
                part_rep_members_gp.append(ens_anom[ifld, i, :] + ens_mean[i, :])

                if i == 0:
                    part_rep_members.append(ifld)
            if i == 0:
                rep_members.append(part_rep_members)
            step_centroids_gp.append(part_centroids_gp)
            step_rep_members_gp.append(part_rep_members_gp)
        centroids_gp.append(step_centroids_gp)
        rep_members_gp.append(step_rep_members_gp)

    return ind_cl, var_opt, centroids, rep_members, centroids_gp, rep_members_gp


def select_optimal_partition(config: ClusterConfig, var_opt: list, noise_var: np.ndarray, sig_thr: np.ndarray, ens_spread: float, verbose: bool = False) -> int:
    """Select the optimal partition

    Parameters
    ----------
    config: ClusterConfig
        Configuration
    var_opt: list (ncl_max+1)
        Variances of the partitions as tuples (centroid variance, internal
        variance, variance ratio)
    noise_var: numpy array (nsamples, ncl_max-1)
        Variance of the red noise samples (first index is number of clusters - 2)
    sig_thr: numpy array (ncl_max + 1)
        Significance thresholds (index is number of clusters)
    ens_spread: float
        Ensemble spread
    verbose: bool (default False)
        If True, print diagnostics

    Returns
    -------
    int
        Optimal number of clusters
    """
    # Compute significance
    sig = np.zeros(config.ncl_max + 1)
    for ncl in range(2, config.ncl_max + 1):
        dsig = 100 / max(1, config.nrsamples)
        var_ratio = var_opt[ncl][2]

        for i in range(config.nrsamples):
            if var_ratio > noise_var[i, ncl-2]:
                sig[ncl] += dsig

        # TODO: write diagnostics?
        if verbose:
            print(f"Significance of the {ncl}-cluster partition: {sig[ncl]}")
            print(f"Variance ratio: {var_ratio}, threshold for significance: {sig_thr[ncl]}")

    # Choose the best partition based on significance
    candidates = [
        ncl for ncl in range(2, config.ncl_max + 1)
        if var_opt[ncl][1] >= ens_spread
            and var_opt[ncl][2] >= sig_thr[ncl]
    ]
    best_ncl = 1
    best_sig = 0.
    for ncl in candidates:
        if best_sig < sig[ncl]:
            best_ncl = ncl
            best_sig = sig[ncl]

            if sig[ncl] >= config.max_sig:
                break

    if best_ncl == config.ncl_max:
        # Try and find a smaller partition
        candidates.remove(config.ncl_max)
        for ncl in candidates[::-1]:
            if sig[ncl] >= config.med_sig:
                best_ncl = ncl
                best_sig = sig[ncl]

    if best_ncl == config.ncl_max and best_sig >= (config.min_sig + config.sig_tol):
        for ncl in candidates[::-1]:
            if sig[ncl] >= (sig[config.ncl_max] - config.sig_tol):
                best_ncl = ncl
                best_sig = sig[ncl]

    return best_ncl


def find_cluster(fields: np.ndarray, ens_mean: np.ndarray, eof: np.ndarray, weights: np.ndarray, centroids: np.ndarray) -> int:
    """Find the cluster containing a time series of fields

    Parameters
    ----------
    fields: numpy array (nsteps, npoints)
        Time steps
    ens_mean: numpy array (nsteps, npoints)
        Ensemble mean
    eof: numpy.ndarray (neof, nsteps, npoints)
        EOFs
    weights: numpy.ndarray (npoints)
        Geographical weights
    centroids: numpy.ndarray (nclusters, neof)
        Cluster centroids

    Returns
    -------
    int
        Cluster index
    """
    norm = np.sqrt(np.einsum('ijk,ijk,k->i', eof, eof, weights))
    fields_proj = np.einsum('jk,ijk,k->i', fields - ens_mean, eof, weights) / norm
    dist2 = np.sum(np.square(fields_proj[np.newaxis, :] - centroids), axis=1)
    return np.argmin(dist2)


def get_output_keys(config: ClusterConfig, template: GRIBMessage) -> dict:
    """Construct the dictionary of GRIB keys to set on the output files"""
    keys = dict(
        clusteringMethod=4,
        startTimeStep=config.step_start,
        endTimeStep=config.step_end,
        northernLatitudeOfDomain=int(config.lat_n * 1000),
        southernLatitudeOfDomain=int(config.lat_s * 1000),
        westernLongitudeOfDomain=int(config.lon_w * 1000),
        easternLongitudeOfDomain=int(config.lon_e * 1000),
        clusteringDomain='h',
    )

    if config.monthly:
        keys['startTimeStep'] = config.step_start - config.step_del + 24

    extract = [
        'parameter', 'level', 'date', 'time', 'stream', 'Ni', 'Nj',
        'latitudeOfFirstGridPointInDegrees', 'longitudeOfFirstGridPointInDegrees',
        'latitudeOfLastGridPointInDegrees', 'longitudeOfLastGridPointInDegrees',
        'jDirectionIncrementInDegrees', 'iDirectionIncrementInDegrees',
    ]
    for key in extract:
        keys[key] = template[key]

    if config.monthly:
        steps = [(e - config.step_del + 24, e) for e in config.steps]
    else:
        steps = [(s, None) for s in config.steps]


    return keys, steps


def do_clustering(config: ClusterConfig, data: dict, npc: int, verbose: bool = False, dump_indexes = None):
    """Run the ensemble clustering

    Parameters
    ----------
    config: ClusterConfig
        Clustering configuration
    data: dict
        PCA data
    npc: int
        Number of principal components to use
    verbose: bool
        If True, print out diagnostics
    dump_indexes: path-like, optional
        If set, write out the cluster indexes to this file (npz)

    Returns
    -------
    numpy array (nfld)
        Cluster index for each member
    numpy array (ncl, npc)
        Cluster centroids in PC space
    list[int] (ncl)
        Cluster representative member indexes
    list[list[numpy array]] (nsteps, ncl, npoints)
        Cluster centroids in grid point space
    list[list[numpy array]] (nsteps, ncl, npoints)
        Cluster representative members in grid point space
    numpy array (nsteps, npoints)
        Ensemble mean
    """
    pc, eof_sd, ens_spread, ens_anom, ens_mean, pc_sd, pc_ac = prepare_data(data, npc, config.factor, verbose=verbose)

    # Compute thresholds for cluster significance
    sig_thr = compute_variance_thresholds(config.ncl_max, npc, eof_sd, verbose=True)

    # Initialise random number generator
    rand = init_rand(pc)

    # Perform the clustering
    ind_cl, var_opt, centroids, rep_members, centroids_gp, rep_members_gp = compute_clusters(ens_anom, ens_mean, pc, config.ncl_max, config.npass, rand, verbose=verbose)

    # Write out the indexes
    if dump_indexes is not None:
        nfld = ens_anom.shape[0]
        ind_cl[1] = np.zeros(nfld, dtype=int)
        np.savez_compressed(dump_indexes, **{'ind_cl': np.asarray(ind_cl[1:])})

    # Perform a clustering on red noise
    noise_var = red_noise_cluster(config.nrsamples, config.ncl_max, config.npass, npc, nfld, pc_sd, pc_ac, rand, config.n_par)

    # Select optimal partition
    best_ncl = select_optimal_partition(config, var_opt, noise_var, sig_thr, ens_spread, verbose=True)
    if verbose:
        print(f"Optimal partition: {best_ncl} cluster{'s'*(best_ncl>1)}")

    if best_ncl == 1:
        # No better partition than no partition at all
        norm2 = np.sum(np.square(pc), axis=0)
        ifld = np.argmin(norm2)
        rep_members[1] = [ifld]
        ind_cl[1] = np.zeros(nfld, dtype=int)
        centroids[1] = np.mean(pc, axis=1)[np.newaxis, :]
        nstep = ens_anom.shape[1]
        for i in range(nstep):
            centroids_gp[i][1] = [ens_mean[i, :]]
            rep_members_gp[i][1] = [ens_anom[ifld, i, :] + ens_mean[i, :]]

    # Extract selected partition
    ind_cl = ind_cl[best_ncl]
    centroids = centroids[best_ncl]
    rep_members = rep_members[best_ncl]
    centroids_gp = [step[best_ncl] for step in centroids_gp]
    rep_members_gp = [step[best_ncl] for step in rep_members_gp]

    return ind_cl, centroids, rep_members, centroids_gp, rep_members_gp, ens_mean


def get_parser() -> argparse.ArgumentParser:
    """initialize command line application argument parser.

    Returns
    -------
    argparse.ArgumentParser
        
    """

    _description='K-Means clustering of ensemble data'
    parser = default_parser(description=_description)

    group = parser.add_argument_group('Clustering arguments')

    group.add_argument('-d', '--deterministic', default=None, help="Deterministic forecast (GRIB)")
    group.add_argument('-p', '--pca', required=True, help="PCA data (NPZ)")
    group.add_argument('-t', '--template', required=True, help="Field to extract keys from (GRIB)")
    group.add_argument('-C', '--centroids', required=True, help="Cluster centroids output (GRIB)")
    group.add_argument('-R', '--representative', required=True, help="Cluster representative members output (GRIB)")
    group.add_argument('-I', '--indexes', required=True, help="Cluster indexes output (NPZ)")
    group.add_argument('-N', '--ncomp-file', default=None, help="Number of components output (text)")
   
    return parser


def main(args=sys.argv[1:]):
    parser = get_parser()

    args = parser.parse_args(args)

    config = ClusterConfig(args)

    data = np.load(args.pca)

    # Compute number of PCs based on the variance threshold
    var_cum = data['var_cum']
    npc = config.npc
    if npc is None:
        npc = select_npc(config.var_th, var_cum)
        if args.ncomp_file is not None:
            with open(args.ncomp_file, 'w') as f:
                print(npc, file=f)

    print(f"Number of PCs used: {npc}, explained variance: {var_cum[npc-1]} %")

    ind_cl, centroids, rep_members, centroids_gp, rep_members_gp, ens_mean = do_clustering(config, data, npc, verbose=True, dump_indexes=args.indexes)

    # Find the deterministic forecast
    if args.deterministic is not None:
        det = read_steps_grib(args.deterministic, config.steps)
        det_index = find_cluster(det, ens_mean, data['eof'][:npc, ...], data['weights'], centroids)
    else:
        det_index = 0

    # Write output
    with FileReader(args.template) as reader:
        message = next(reader)
        keys, steps = get_output_keys(config, message)

    target = FileTarget(args.centroids)
    keys['type'] = 'cm'
    write_cluster_grib(steps, ind_cl, rep_members, det_index, centroids_gp, target, keys)

    target = FileTarget(args.representative)
    keys['type'] = 'cr'
    write_cluster_grib(steps, ind_cl, rep_members, det_index, rep_members_gp, target, keys)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
