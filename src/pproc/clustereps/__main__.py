# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime, timedelta
from os.path import join as pjoin
import sys
from typing import List

from conflator import Conflator
from meters import ResourceMeter
import numpy as np

import eccodes

from pproc.clustereps import attribution, cluster, pca
from pproc.clustereps.io import read_ensemble_grib, read_steps_grib
from pproc.common.dataset import open_dataset
from pproc.common.io import FDBNotOpenError, fdb
from pproc.config.io import InputsCollection, Output
from pproc.config.types import ClusterFullConfig


def get_mean_spread(
    inputs: InputsCollection,
    src_name: str,
    date: datetime,
    steps: List[int],
    ndays: int = 31,
) -> np.ndarray:
    """Compute mean spread over the last days

    Parameters
    ----------
    inputs: InputsCollection
        Inputs configuration
    src_name: str
        Source name
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
        data = read_steps_grib(
            inputs, src_name, steps, date=ret_date.strftime("%Y%m%d")
        )
        print(f"{ret_date:%Y%m%d} {data.shape!s:20s} {data.min():15g} {data.max():15g}")
        nfields += data.shape[0]
        if spread is None:
            spread = np.sum(data, axis=0)
        else:
            spread += np.sum(data, axis=0)
    assert spread is not None
    spread /= nfields
    return spread


def write_cluster_attr_grib(
    steps,
    ind_cl,
    rep_members,
    det_index,
    data,
    anom_data,
    cluster_att,
    target,
    anom_target,
    keys,
    ncl_dummy=None,
):
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
    sample.set("totalNumberOfClusters", ncl)
    sample.set("controlForecastCluster", ind_cl[0] + 1)
    for icl in range(ncl):
        members = np.nonzero(ind_cl == icl)[0]

        message = sample.copy()
        message.set("clusterNumber", icl + 1)
        message.set("numberOfForecastsInCluster", len(members))
        message.set_array("ensembleForecastNumbers", members)
        message.set("operationalForecastCluster", det_index + 1)
        message.set("representativeMember", rep_members[icl])

        for i, (start, end) in enumerate(steps):
            if end is None:
                message.set("step", start)
            else:
                message.set("startStep", start)
                message.set("endStep", end)
                message.set("stepRange", f"{start}-{end}")

            message.set("climatologicalRegime", cluster_att[i, icl])

            message.set_array("values", data[i, icl])
            target.write(message)

            message.set_array("values", anom_data[i, icl])
            anom_target.write(message)

    if ncl_dummy is None:
        return

    sample.set("clusteringMethod", 0)
    sample.set("numberOfForecastsInCluster", 0)
    sample.set("operationalForecastCluster", 0)
    sample.set("representativeMember", 0)
    sample.set("climatologicalRegime", 0)
    sample.set("controlForecastCluster", 0)
    dummy_data = np.zeros_like(data[0, 0])
    sample.set_array("values", dummy_data)
    for icl in range(ncl, ncl_dummy):
        message = sample.copy()
        message.set("clusterNumber", icl + 1)

        for i, (start, end) in enumerate(steps):
            if end is None:
                message.set("step", start)
            else:
                message.set("startStep", start)
                message.set("endStep", end)
                message.set("stepRange", f"{start}-{end}")

            target.write(message)
            anom_target.write(message)


def main():
    sys.stdout.reconfigure(line_buffering=True)

    cfg: ClusterFullConfig = Conflator(
        app_name="pproc-clustereps", model=ClusterFullConfig
    ).load()
    cfg.print()

    # PCA

    ## Read or compute ensemble stddev
    with ResourceMeter("Ensemble spread"):
        if cfg.compute_spread:
            spread = get_mean_spread(cfg.inputs, "spread", cfg.date, cfg.steps)
        else:
            spread_source = cfg.inputs.spread
            with open_dataset(
                spread_source.legacy_config(), spread_source.location()
            ) as reader:
                message = next(reader)
                # TODO: check param and level
                spread = message.get_array("values")

    ## Read ensemble
    with ResourceMeter("Read ensemble"):
        nexp = cfg.num_members
        lat, lon, ens, grib_template = read_ensemble_grib(
            cfg.inputs, "fc", cfg.steps, nexp
        )

    ## Compute PCA
    with ResourceMeter("PCA"):
        pca_data = pca.do_pca(cfg, lat, lon, ens, spread)

    ## Save data
    if cfg.pca_output is not None:
        np.savez_compressed(cfg.pca_output, **pca_data)

    # Clustering

    ## Compute number of PCs based on the variance threshold
    var_cum = pca_data["var_cum"]
    npc = cfg.npc
    if npc <= 0:
        npc = cluster.select_npc(cfg.var_th, var_cum)
        if cfg.ncomp_file is not None:
            with open(cfg.ncomp_file, "w") as f:
                print(npc, file=f)

    print(f"Number of PCs used: {npc}, explained variance: {var_cum[npc-1]} %")

    with ResourceMeter("Clustering"):
        ind_cl, centroids, rep_members, centroids_gp, rep_members_gp, ens_mean = (
            cluster.do_clustering(
                cfg, pca_data, npc, verbose=True, dump_indexes=cfg.indexes
            )
        )

    ## Find the deterministic forecast
    if cfg.deterministic_is_control:
        det_index = ind_cl[0]
    elif cfg.inputs.deterministic.type != "null":
        with ResourceMeter("Find deterministic"):
            det = read_steps_grib(cfg.inputs, "deterministic", cfg.steps)
            det_index = cluster.find_cluster(
                det,
                ens_mean,
                pca_data["eof"][:npc, ...],
                pca_data["weights"],
                centroids,
            )
    else:
        det_index = 0

    # Attribution

    cluster_data = {
        "centroids": centroids_gp,
        "representative": rep_members_gp,
    }
    cluster_dests: dict[str, tuple[Output, Output]] = {
        "centroids": (cfg.outputs.centroids, cfg.outputs.cen_anomalies),
        "representative": (cfg.outputs.representative, cfg.outputs.rep_anomalies),
    }

    with ResourceMeter("Read climatology"):
        ## Read climatology fields
        clim = attribution.get_climatology_fields(
            cfg.clim_means, cfg.seasons, cfg.step_date
        )

        ## Read climatological EOFs
        clim_eof, clim_ind = attribution.get_climatology_eof(
            cfg.clim_cluster_centroids_eof,
            cfg.clim_eof,
            cfg.clim_pcs,
            cfg.clim_sdv,
            cfg.clim_cluster_index,
            cfg.ncl_clim,
            cfg.month_start_dos,
            cfg.month_end_dos,
        )

    keys, steps = cluster.get_output_keys(cfg, grib_template)

    for scenario, scdata in cluster_data.items():
        scdata = np.array(scdata)
        weights = pca_data["weights"]

        ## Compute anomalies
        anom = scdata - clim
        anom = np.clip(anom, -cfg.max_anom, cfg.max_anom)

        with ResourceMeter(f"Attribute {scenario}"):
            cluster_att, min_dist = attribution.attribution(
                anom, clim_eof, clim_ind, weights
            )

        with ResourceMeter(f"Write {scenario} output"):
            ## Write anomalies and cluster scenarios
            dest, adest = cluster_dests[scenario]
            write_cluster_attr_grib(
                steps,
                ind_cl,
                rep_members,
                det_index,
                scdata,
                anom,
                cluster_att,
                dest.target,
                adest.target,
                {**dest.metadata, **keys, **cfg.metadata},
                ncl_dummy=cfg.ncl_dummy,
            )

            ## Write report output
            # table: attribution cluster index for all fc clusters, step
            np.savetxt(
                pjoin(
                    cfg.output_root,
                    f"{cfg.step_start}_{cfg.step_end}dist_index_{scenario}.txt",
                ),
                min_dist,
                fmt="%-10.5f",
                delimiter=3 * " ",
            )

            # table: distance measure for all fc clusters, step
            np.savetxt(
                pjoin(
                    cfg.output_root,
                    f"{cfg.step_start}_{cfg.step_end}att_index_{scenario}.txt",
                ),
                cluster_att,
                fmt="%-3d",
                delimiter=3 * " ",
            )

    try:
        fdb(create=False).flush()
    except FDBNotOpenError:
        pass
    cfg.clean()

    return 0


if __name__ == "__main__":
    exit(main())
