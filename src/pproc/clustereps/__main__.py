
import argparse
from datetime import datetime
import os
from os.path import join as pjoin

import numpy as np

import eccodes

from pproc.clustereps import attribution, cluster, pca
from pproc.clustereps.io import open_dataset, read_ensemble_grib, read_steps_grib
from pproc.common import default_parser


def get_parser() -> argparse.ArgumentParser:
    """Initialize the command-line argument parser

    Returns
    -------
    argparse.ArgumentParser
    """

    _description='K-Means clustering of ensemble data'
    parser = default_parser(description=_description)

    group = parser.add_argument_group('General arguments')
    group.add_argument('--date', help='forecast date (YYMMDD)', type=lambda x: datetime.strptime(x, '%Y%m%d'), metavar='YMD')

    group = parser.add_argument_group('Principal component analysis arguments')
    group.add_argument('-m', '--mask', default=None, help="Mask file")
    group.add_argument('--spread', required=True, help="Ensemble spread (GRIB)")
    group.add_argument('-e', '--ensemble', required=True, help="Ensemble data (GRIB)")
    group.add_argument('-P', '--pca', default=None, help="Output file (NPZ)")

    group = parser.add_argument_group('Clustering arguments')
    group.add_argument('-d', '--deterministic', default=None, help="Deterministic forecast (GRIB)")
    group.add_argument('-C', '--centroids', default=None, help="Cluster centroids output (GRIB)")
    group.add_argument('-R', '--representative', default=None, help="Cluster representative members output (GRIB)")
    group.add_argument('-I', '--indexes', default=None, help="Cluster indexes output (NPZ)")
    group.add_argument('-N', '--ncomp-file', default=None, help="Number of components output (text)")

    group = parser.add_argument_group('Attribution arguments')
    group.add_argument('--clim-dir', help='climatological data root directory', metavar='DIR')
    group.add_argument('-o', '--output-root', default=os.getcwd(), help='output base directory', metavar='DIR')

    return parser


def write_cluster_attr_grib(steps, ind_cl, rep_members, det_index, data, anom_data, cluster_att, target, anom_target, keys):
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


def main(sys_args=None):
    parser = get_parser()
    args = parser.parse_args(sys_args)

    # PCA

    pca_config = pca.PCAConfig(args)

    ## Read mask
    if args.mask is not None:
        # format?
        raise NotImplementedError()

    ## Read ensemble
    nexp = pca_config.num_members
    lat, lon, ens, grib_template = read_ensemble_grib(pca_config.sources, args.ensemble, pca_config.steps, nexp)

    ## Read ensemble stddev
    with open_dataset(pca_config.sources, args.spread) as reader:
        message = next(reader)
        # TODO: check param and level
        spread = message.get_array('values')

    ## Compute PCA
    pca_data = pca.do_pca(pca_config, lat, lon, ens, spread, args.mask)

    ## Save data
    if args.pca is not None:
        np.savez_compressed(args.pca, **pca_data)

    # Clustering

    cluster_config = cluster.ClusterConfig(args, verbose=False)

    ## Compute number of PCs based on the variance threshold
    var_cum = pca_data['var_cum']
    npc = cluster_config.npc
    if npc <= 0:
        npc = cluster.select_npc(cluster_config.var_th, var_cum)
        if args.ncomp_file is not None:
            with open(args.ncomp_file, 'w') as f:
                print(npc, file=f)

    print(f"Number of PCs used: {npc}, explained variance: {var_cum[npc-1]} %")

    ind_cl, centroids, rep_members, centroids_gp, rep_members_gp, ens_mean = cluster.do_clustering(cluster_config, pca_data, npc, verbose=True, dump_indexes=args.indexes)

    ## Find the deterministic forecast
    if args.deterministic is not None:
        det = read_steps_grib(cluster_config.sources, args.deterministic, cluster_config.steps)
        det_index = cluster.find_cluster(det, ens_mean, pca_data['eof'][:npc, ...], pca_data['weights'], centroids)
    else:
        det_index = 0

    ## Write output
    keys, steps = cluster.get_output_keys(cluster_config, grib_template)

    if args.centroids is not None:
        target = cluster.FileTarget(args.centroids)
        keys['type'] = 'cm'
        cluster.write_cluster_grib(steps, ind_cl, rep_members, det_index, centroids_gp, target, keys)

    if args.representative is not None:
        target = cluster.FileTarget(args.representative)
        keys['type'] = 'cr'
        cluster.write_cluster_grib(steps, ind_cl, rep_members, det_index, rep_members_gp, target, keys)

    # Attribution

    attr_config = attribution.AttributionConfig(args, verbose=False)

    cluster_data = {
        'centroids': centroids_gp,
        'representative': rep_members_gp,
    }
    cluster_types = {
        'centroids': 'cm',
        'representative': 'cr',
    }

    ## Read climatology fields
    clim = attribution.get_climatology_fields(
        attr_config.climMeans, attr_config.seasons, attr_config.stepDate
    )

    ## Read climatological EOFs
    clim_eof, clim_ind = attribution.get_climatology_eof(
        attr_config.climClusterCentroidsEOF,
        attr_config.climEOFs,
        attr_config.climPCs,
        attr_config.climSdv,
        attr_config.climClusterIndex,
        attr_config.nClusterClim,
        attr_config.monStartDoS,
        attr_config.monEndDoS,
    )

    for scenario, scdata in cluster_data.items():
        scdata = np.array(scdata)
        weights = pca_data['weights']

        ## Compute anomalies
        anom = scdata - clim
        anom = np.clip(anom, -attr_config.clip, attr_config.clip)

        cluster_att, min_dist = attribution.attribution(anom, clim_eof, clim_ind, weights)

        ## Write anomalies and cluster scenarios
        target = cluster.FileTarget(
            pjoin(attr_config.output_root, f'{attr_config.step_start}_{attr_config.step_end}{scenario}.grib')
        )
        anom_target = cluster.FileTarget(
            pjoin(attr_config.output_root, f'{attr_config.step_start}_{attr_config.step_end}{scenario}_anom.grib')
        )
        keys['type'] = cluster_types[scenario]
        write_cluster_attr_grib(steps, ind_cl, rep_members, det_index, scdata, anom, cluster_att, target, anom_target, keys)

        ## Write report output
        # table: attribution cluster index for all fc clusters, step
        np.savetxt(
            pjoin(attr_config.output_root, f'{attr_config.step_start}_{attr_config.step_end}dist_index_{scenario}.txt'), min_dist,
            fmt='%-10.5f', delimiter=3*' '
        )

        # table: distance measure for all fc clusters, step
        np.savetxt(
            pjoin(attr_config.output_root, f'{attr_config.step_start}_{attr_config.step_end}att_index_{scenario}.txt'), cluster_att,
            fmt='%-3d', delimiter=3*' '
        )

    return 0


if __name__ == '__main__':
    exit(main())