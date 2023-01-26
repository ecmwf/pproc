
import argparse
from datetime import datetime, timedelta
import os
from os.path import join as pjoin
from typing import List

import numpy as np

import eccodes

from pproc.clustereps import attribution, cluster, pca
from pproc.clustereps.config import FullClusterConfig
from pproc.clustereps.io import open_dataset, read_ensemble_grib, read_steps_grib, target_from_location
from pproc.common import default_parser


def get_mean_spread(sources: dict, locs: List[str], date: datetime, steps: List[int], ndays: int = 31) -> np.ndarray:
    """Compute mean spread over the last days

    Parameters
    ----------
    sources: dict
        Sources configuration
    locs: list[str]
        Locations of the data (file path, named fdb request, ...)
    date: datetime
        Reference date (not included in the timespan)
    steps: list[int]
        Time steps to accumulate
    ndays: int
        Number of days to accumulate (date - ndays, ..., date - 1)

    Returns
    -------
    numpy array (npoints)
        Mean spread over all dates and steps
    """
    spread = None
    nfields = 0
    for diff in range(ndays, 0, -1):
        ret_date = date - timedelta(days=diff)
        found = False
        for loc in locs:
            try:
                data = read_steps_grib(sources, loc, steps, date=ret_date.strftime("%Y%m%d"))
            except (EOFError, eccodes.IOProblemError, FileNotFoundError):
                continue
            print(f"{ret_date:%Y%m%d} {loc:20s} {data.shape!s:20s} {data.min():15g} {data.max():15g}")
            nfields += data.shape[0]
            if spread is None:
                spread = np.sum(data, axis=0)
            else:
                spread += np.sum(data, axis=0)
            found = True
            break
        if not found:
            raise ValueError(f"Could not find data for date {ret_date:%Y%m%d}")
    assert spread is not None
    spread /= nfields
    return spread


def get_parser() -> argparse.ArgumentParser:
    """Initialize the command-line argument parser

    Returns
    -------
    argparse.ArgumentParser
    """

    _description='K-Means clustering of ensemble data'
    parser = default_parser(description=_description)

    group = parser.add_argument_group('General arguments')
    group.add_argument('--date', help="Forecast date (YYMMDD)", type=lambda x: datetime.strptime(x, '%Y%m%d'), metavar='YMD')

    group = parser.add_argument_group('Inputs')
    group.add_argument('-m', '--mask', default=None, help="Mask file")

    sgroup = group.add_mutually_exclusive_group(required=True)
    sgroup.add_argument('--spread', default=None, help="Ensemble spread (GRIB)")
    sgroup.add_argument('--spread-compute', action="append", help="Source for ensemble spread computation (GRIB)")

    group.add_argument('-e', '--ensemble', required=True, help="Ensemble data (GRIB)")
    group.add_argument('-d', '--deterministic', default=None, help="Deterministic forecast (GRIB)")
    group.add_argument('--clim-dir', help="Climatological data root directory", metavar='DIR')

    group = parser.add_argument_group('Outputs')
    group.add_argument('-P', '--pca', default=None, help="PCA outputs (NPZ)")
    group.add_argument('-C', '--centroids', default=None, help="Cluster centroids output (GRIB)")
    group.add_argument('-R', '--representative', default=None, help="Cluster representative members output (GRIB)")
    group.add_argument('-CA', '--cen-anomalies', default=None, help="Cluster centroids output in anomaly space (GRIB)")
    group.add_argument('-RA', '--rep-anomalies', default=None, help="Cluster representative members output in anomaly space (GRIB)")
    group.add_argument('-I', '--indexes', default=None, help="Cluster indexes output (NPZ)")
    group.add_argument('-N', '--ncomp-file', default=None, help="Number of components output (text)")
    group.add_argument('-o', '--output-root', default=os.getcwd(), help="Output directory for reports", metavar='DIR')

    return parser


def write_cluster_attr_grib(steps, ind_cl, rep_members, det_index, data, anom_data, cluster_att, target, anom_target, keys, ncl_dummy=None):
    """Write attributed clustering data to a GRIB output

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
    anom_data: numpy array(nsteps, ncl, ngp)
        Data to write in anomaly grid point space
    target: any object with a ``write(eccodes.Message)`` method
        Write target
    anom_target: any object with a ``write(eccodes.Message)`` method
        Write target for anomalies
    keys: dict-like
        GRIB keys to set
    ncl_dummy: int, optional
        If set, generate placeholders for clusters ncl+1, ..., dummy_clusters
    """
    ncl = len(rep_members)
    sample = eccodes.GRIBMessage.from_samples("clusters_grib1").copy()
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

            message.set('climatologicalRegime', cluster_att[i, icl])

            message.set_array('values', data[i, icl])
            target.write(message)

            message.set_array('values', anom_data[i, icl])
            anom_target.write(message)

    if ncl_dummy is None:
        return

    sample.set('clusteringMethod', 0)
    sample.set('numberOfForecastsInCluster', 0)
    sample.set('operationalForecastCluster', 0)
    sample.set('representativeMember', 0)
    sample.set('climatologicalRegime', 0)
    sample.set('controlForecastCluster', 0)
    dummy_data = np.zeros_like(data[0, 0])
    sample.set_array('values', dummy_data)
    for icl in range(ncl, ncl_dummy):
        message = sample.copy()
        message.set('clusterNumber', icl + 1)

        for i, (start, end) in enumerate(steps):
            if end is None:
                message.set('step', start)
            else:
                message.set('startStep', start)
                message.set('endStep', end)
                message.set('stepRange', f"{start}-{end}")

            target.write(message)
            anom_target.write(message)


def main(sys_args=None):
    parser = get_parser()
    args = parser.parse_args(sys_args)

    config = FullClusterConfig(args)

    # PCA

    ## Read or compute ensemble stddev
    if args.spread is not None:
        with open_dataset(config.sources, args.spread) as reader:
            message = next(reader)
            # TODO: check param and level
            spread = message.get_array('values')
    else:
        spread = get_mean_spread(config.sources, args.spread_compute, args.date, config.steps)

    ## Read mask
    if args.mask is not None:
        # format?
        raise NotImplementedError()

    ## Read ensemble
    nexp = config.num_members
    lat, lon, ens, grib_template = read_ensemble_grib(config.sources, args.ensemble, config.steps, nexp)

    ## Compute PCA
    pca_data = pca.do_pca(config, lat, lon, ens, spread, args.mask)

    ## Save data
    if args.pca is not None:
        np.savez_compressed(args.pca, **pca_data)

    # Clustering

    ## Compute number of PCs based on the variance threshold
    var_cum = pca_data['var_cum']
    npc = config.npc
    if npc <= 0:
        npc = cluster.select_npc(config.var_th, var_cum)
        if args.ncomp_file is not None:
            with open(args.ncomp_file, 'w') as f:
                print(npc, file=f)

    print(f"Number of PCs used: {npc}, explained variance: {var_cum[npc-1]} %")

    ind_cl, centroids, rep_members, centroids_gp, rep_members_gp, ens_mean = cluster.do_clustering(config, pca_data, npc, verbose=True, dump_indexes=args.indexes)

    ## Find the deterministic forecast
    if args.deterministic is not None:
        det = read_steps_grib(config.sources, args.deterministic, config.steps)
        det_index = cluster.find_cluster(det, ens_mean, pca_data['eof'][:npc, ...], pca_data['weights'], centroids)
    else:
        det_index = 0

    # Attribution

    cluster_data = {
        'centroids': centroids_gp,
        'representative': rep_members_gp,
    }
    cluster_types = {
        'centroids': 'cm',
        'representative': 'cr',
    }
    cluster_dests = {
        'centroids': (args.centroids, args.cen_anomalies),
        'representative': (args.representative, args.rep_anomalies),
    }

    ## Read climatology fields
    clim = attribution.get_climatology_fields(
        config.climMeans, config.seasons, config.stepDate
    )

    ## Read climatological EOFs
    clim_eof, clim_ind = attribution.get_climatology_eof(
        config.climClusterCentroidsEOF,
        config.climEOFs,
        config.climPCs,
        config.climSdv,
        config.climClusterIndex,
        config.nClusterClim,
        config.monStartDoS,
        config.monEndDoS,
    )

    keys, steps = cluster.get_output_keys(config, grib_template)

    for scenario, scdata in cluster_data.items():
        scdata = np.array(scdata)
        weights = pca_data['weights']

        ## Compute anomalies
        anom = scdata - clim
        anom = np.clip(anom, -config.clip, config.clip)

        cluster_att, min_dist = attribution.attribution(anom, clim_eof, clim_ind, weights)

        ## Write anomalies and cluster scenarios
        dest, adest = cluster_dests[scenario]
        target = target_from_location(dest)
        anom_target = target_from_location(adest)
        keys['type'] = cluster_types[scenario]
        write_cluster_attr_grib(steps, ind_cl, rep_members, det_index, scdata, anom, cluster_att, target, anom_target, keys, ncl_dummy=config.ncl_dummy)

        ## Write report output
        # table: attribution cluster index for all fc clusters, step
        np.savetxt(
            pjoin(config.output_root, f'{config.step_start}_{config.step_end}dist_index_{scenario}.txt'), min_dist,
            fmt='%-10.5f', delimiter=3*' '
        )

        # table: distance measure for all fc clusters, step
        np.savetxt(
            pjoin(config.output_root, f'{config.step_start}_{config.step_end}att_index_{scenario}.txt'), cluster_att,
            fmt='%-3d', delimiter=3*' '
        )

    return 0


if __name__ == '__main__':
    exit(main())