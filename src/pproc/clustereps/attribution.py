import sys
import argparse
from datetime import datetime, timedelta
from calendar import isleap
from typing import Iterator, List, Tuple
import os
from os.path import join as pjoin

import numpy as np
from scipy.io import FortranFile

from eccodes import FileReader

from pproc.clustereps.utils import region_weights, lat_weights, normalise_angles
from pproc.common import Config, default_parser


MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


class Season:

    def __init__(self, startMonth, endMonth, baseYear):
        
        self.baseYear = baseYear
        jumpYear = 0
        if startMonth > endMonth:
            startYear = self.baseYear - 1
            jumpYear = 1
        else:
            startYear = self.baseYear
        first_day = datetime.strptime(f"{startYear:04d}{startMonth:02d}01", "%Y%m%d")
        end_day = datetime.strptime(f"{self.baseYear:04d}{endMonth:02d}{MONTH_DAYS[endMonth-1]}", "%Y%m%d")
        
        self.start = first_day
        self.end = end_day

        all_months = list(range(1, 13)) * 2
        self.months = all_months[startMonth - 1 : (endMonth + jumpYear*12)]

        self.name = ''.join([
            datetime.strptime(f"1970{mon:02d}01", "%Y%m%d").strftime("%b").lower()[0] for mon in self.months
        ])

    @property
    def ndays(self) -> int:
        return (self.end - self.start).days + 1

    @property
    def doys(self) -> List[int]:
        return [(self.start + timedelta(days=i)).timetuple().tm_yday - 1 for i in range(self.ndays)]

    def dos(self, date: datetime) -> int:
        return (date - self.start).days

    def __len__(self) -> int:
        return len(self.months)

    def __contains__(self, date: datetime) -> bool:
        return self.start.date() <= date.date() <= self.end.date()

    def __repr__(self) -> str:
        return f"Season ({self.name}) - {self.start:%d/%m/%Y}: {self.end:%d/%m/%Y} ({self.ndays} days)"


class SeasonConfig:
    def __init__(self, months: List[Tuple[int]]):
        self.months = months

    def get_season(self, date: datetime) -> Season:
        month = date.month
        for start, end in self.months:
            if start <= month <= end:
                return Season(start, end, date.year)
            if end < start:
                if month <= end:
                    return Season(start, end, date.year)
                if start <= month:
                    return Season(start, end, date.year + 1)
        raise ValueError(f"No season containing month {month}")


class AttributionConfig(Config):
    def __init__(self, args: argparse.ArgumentParser) -> None:

        super().__init__(args)

        # yaml config file
        self.stepStart = self.options.get('stepStart', 0)
        self.stepEnd = self.options.get('stepEnd', 120)
        self.stepDel = self.options.get('stepDel', 12)
        # main grid specs
        self.nLat = self.options['nLat']
        self.nLon = self.options['nLon']
        # EOF grid dimensions
        self.rLatN = self.options.get('rLatN', 90.)
        self.rLatS = self.options.get('rLatS', -90.)
        self.rLonW = self.options.get('rLonW', 0.)
        self.rLonE = self.options.get('rLonE', 360.)
        # forecast options
        self.nEns = self.options.get('nEns', 51)
        # Climatological data options
        self.nClusterClim = self.options.get('nClusterClim', 6)
        self.climMeans = pjoin(args.clim_dir, self.options['climMeans'])
        self.climPCs = pjoin(args.clim_dir, self.options['climPCs'])
        self.climSdv = pjoin(args.clim_dir, self.options['climSdv'])
        self.climEOFs = pjoin(args.clim_dir, self.options['climEOFs'])
        self.climClusterIndex = pjoin(args.clim_dir, self.options['climClusterIndex'])
        self.climClusterCentroidsEOF = pjoin(args.clim_dir, self.options['climClusterCentroidsEOF'])
        # pca options
        self.anMax = self.options.get('anMax', 10000.)
        self._seasons = self.options.get('seasons', [(1, 12)])
        
        # forecast date
        self.date = args.date
        self.year = self.date.year
        self.month = self.date.month

        self.nSteps = 1 + (self.stepEnd - self.stepStart) // self.stepDel
        self.stepDate = self.date + timedelta(hours=self.stepStart)

        # out directory
        self.output_root = args.output_root

        # seasons parsing
        seasonConfig = SeasonConfig(self._seasons)
        self.seasons = seasonConfig

        self.thisSeason = seasonConfig.get_season(self.date)
        self.monStartDoS = self.thisSeason.dos(datetime(self.year, self.month, 1))
        self.monEndDoS = self.thisSeason.dos(datetime(self.year, self.month, MONTH_DAYS[self.month - 1]))

        # parse clim file if template
        for attr in ['climPCs', 'climSdv', 'climEOFs', 'climClusterIndex', 'climClusterCentroidsEOF']:
            val = getattr(self, attr)
            newVal = val.format(season=self.thisSeason.name)
            setattr(self, attr, newVal)
        
        # config file derived parameters
        # grid specs
        self.nPoints = self.nLon * self.nLat

        self.bbox = (self.rLatN, self.rLatS, self.rLonW, self.rLonE)


    def __repr__(self) -> str:
        return self.__dict__.__repr__()


def get_parser() -> argparse.ArgumentParser:
    """initialize command line application argument parser.

    Returns
    -------
    argparse.ArgumentParser
        
    """

    _description='Cluster attribution to closest partition of climatological weather regimes'
    parser = default_parser(description=_description)

    group = parser.add_argument_group('Attribution arguments')
    group.add_argument('--date', help='forecast date (YYMMDD)', type=lambda x: datetime.strptime(x, '%Y%m%d'), metavar='YMD')
    group.add_argument('--centroids', help='Forecast cluster centroids (GRIB)', metavar='FILENAME.grib')
    group.add_argument('--representative', help='Forecast cluster representative (GRIB)',  metavar='FILENAME.grib')
    group.add_argument('--clim_dir', help='climatological data root directory', metavar='DIR')
    group.add_argument('-o', '--output_root', default=os.getcwd(), help='output base directory', metavar='DIR')

    return parser


def read_grib_cluster(inFile: str, stepStart: int, stepDelta: int, nSteps: int, nEns: int) -> Tuple[np.ndarray]:
    """Read clustering data from a GRIB output

    Parameters
    ----------
    inFile: any object with a ``write(eccodes.Message)`` method
        Write target
    stepStart: integer
        field start timestamp, in hours.
        
    stepDelta: integer
        field increment, in hours.

    nSteps: integer
        number of steps in clustering data.

    nEns: integer
        total number of ensembles in forecast. Will define the dimension of
        second output
    
    Returns
    -------
    cents: numpy.Array(nSteps, nClusters, nPoints) [float64]
        field data read from grib.
    ens_numbers: numpy.Array (nClusters, nEns) [int16]
        ensemble numbers within clusters.
    lat: numpy.Array (nPoints) [float64]
        latitude values
    lon: numpy.Array (nPoints) [float64]
        longitude values

    """
    
    with FileReader(inFile) as reader:
        for imes, message in enumerate(reader):
            # TODO: create check_message func 
            #   param, ilevel, startTimeStep, endTimeStep, northernLatitudeOfDomain, southernLatitudeOfDomain,westernLongitudeOfDomain, easternLongitudeOfDomain
            if imes == 0:
                nclusters = message.get('totalNumberOfClusters')
                npoints = message.get_size('values')
                cents = np.empty((nSteps, nclusters, npoints)).astype('float64')
                ens_numbers = np.empty((nclusters, nEns)).astype('int16')
                lat = message.get('latitudes')
                lon = message.get('longitudes')
            icl = message.get('clusterNumber') - 1
            jstep = (message.get('step') - stepStart) // stepDelta
            # get field values
            cents[jstep, icl, :] = message.get_array('values')
            # get ensemble number in cluster
            nFcsts = message.get('numberOfForecastsInCluster')
            ens_numbers[icl, :nFcsts] = message.get_array('ensembleForecastNumbers')
    
    return cents, ens_numbers, lat, lon


def read_clim_file(filePath: str, nRecords: int, dtype: str ='>f4'):

    arr = list()
    with FortranFile(filePath, 'r', header_dtype='>u4') as fin:
        for _ in range(nRecords):
            arr.append(fin.read_record(dtype))
    return np.vstack(arr) if len(arr) > 1 else np.array(arr[0])


def get_climatology_fields(fname_template: str, seasons: SeasonConfig, date: datetime):
    """Read the climatological field associated to the date according to its season

    Parameters
    ----------
    fname_template : str
        format string for the data file, {season} is replaced by the season name
    seasons : SeasonConfig
        season definitions
    date : datetime
        requested date

    Returns
    -------
    np.Array (nPoints)
        field array values from GRIB
    """
    sea = seasons.get_season(date)
    sea_clim = read_clim_file(fname_template.format(season=sea.name), sea.ndays)
    return sea_clim[sea.dos(date), :]


def get_climatology_eof(
    climClusterInfo:str,
    climEOFs:str,
    climPCs:str,
    climSdv:str,
    climIndex:str,
    nClusters:int,
    monStart:int,
    monEnd:int,
):
    """read climatological static datasets and produce monthly mean principal components.

    Parameters
    ----------
    climClusterInfo : str
        Climatological clustering info file. Should contain number of significant directions to use, number of years, and number of days in season used in the clustering process.
    climEOFs : str
        Climatological clustering EOF file path.
    climPCs : str
        Climatological clustering PCs file path.
    climSdv : str
        Climatological clustering Stadandard deviation file path.
    climIndex : str
        Climatological unique IDs of weather regimes.
    nClusters : int
        Number of clusters used in the climatological clustering.
    monStart : int
        Start of month in day of season.
    monEnd : int
        End of month in day of season.

    Returns
    -------
    np.Array (nEOF, nPoints)
        Climatological EOFs
    np.ndarray (nClusters, nEOF)
        Mean monthly principal components for each cluster and EOF
    """

    grid_specs_file = climClusterInfo
    neof, nyrs, ndays = read_clim_file(grid_specs_file, 1, ">u4")

    #assert ndays == season.ndays, "climatology file must match season definition"

    eof = read_clim_file(climEOFs, neof+1)
    eof = eof[1:, :]  # drop mask

    pcs = read_clim_file(climPCs, nyrs*ndays)
    pcs = pcs[:, :neof]
    pcs = pcs.reshape(nyrs, ndays, neof)

    sdv = read_clim_file(climSdv, 1)[:neof]

    # Compute non-standardized PCs and PCs total variance
    pcs = pcs * sdv
    pcs = np.moveaxis(pcs, -1, 0)

    clus_index = read_clim_file(climIndex, ndays*nyrs)
    clus_index = clus_index[:, nClusters-2]
    clus_index = clus_index.reshape(nyrs, ndays)

    # compute cluster centroids in EOF space for the corresponding month.
    # ? output?
    month_ind = (clus_index[:, monStart: monEnd + 1]).flatten().astype(np.int16)
    clcases = np.bincount(
        month_ind,
        weights=np.ones_like(month_ind).astype(np.int8),
        minlength=nClusters
    )
    clcases = clcases[1:]  # drop cluster 0

    monDays = monEnd - monStart + 1
    mon_pcs = pcs[:, :, monStart: monEnd + 1]
    mon_pcs = mon_pcs.reshape(neof, monDays*nyrs)
    mon_pcs = np.apply_along_axis(
        lambda x: np.bincount(month_ind, weights=x, minlength=nClusters),
        1,
        mon_pcs
    )
    mon_pcs = mon_pcs[:, 1:]  # drop cluster 0
    mon_pcs /= clcases

    mon_pcs = np.nan_to_num(mon_pcs, nan=0.)
    mon_pcs = np.moveaxis(mon_pcs, 0, 1)

    return eof, mon_pcs


def attribution(
    fcField: np.array,
    climEOF: np.array,
    climIndex: np.array,
    weights: np.array,    
):
    """Attribute fields to climatological clusters

    Parameters
    ----------
    fcField : np.Array (nSteps, nClusters, nPoints)
        cluster scenarios in anomaly space
    climEOF : np.Array (nEOF, nPoints)
        climatological EOFs
    climIndex : np.Array (nClusterClim, nEOF)
        mean monthly principal components for each cluster and EOF
    weights : np.Array (nPoints)
        geometry-based weights

    Returns
    -------
    np.Array (nSteps, nClusters) [int]
        closest climatological cluster index (1-based)
    np.Array (nSteps, nClusters)
        RMS distance between the projected field and the closest climatological cluster
    """

    # ? Check on clim eof grid? in fortran there is a "change_grid" routine but it's never triggered and ngpeof is forcelly set to ngp in frame_attribute executable
    # 3.3) project anomalies onto climatological eof
    norm = np.sqrt(np.einsum('ij,ij,j->i', climEOF, climEOF, weights))
    anom_proj = np.einsum('zijk,jk,k->zij', fcField[:, :, np.newaxis, :], climEOF, weights) / norm

    #del eof

    # 3.4) Compute distance between clusters and weather regimes
    diff = anom_proj[:, :, np.newaxis, :] - climIndex[np.newaxis, np.newaxis, :, :]
    rms = np.sqrt((diff ** 2).mean(axis=-1))
    min_dist = rms.min(axis=-1)
    cluster_index = rms.argmin(axis=-1) + 1

    return cluster_index, min_dist


def main(sysArgs: List = sys.argv[1:]) -> int:

    parser = get_parser()

    args = parser.parse_args(sysArgs)

    config = AttributionConfig(args)

    clusterFiles = {
        'centroids': args.centroids,
        'representative': args.representative,
    }

    # read climatology fields
    clim = get_climatology_fields(
        config.climMeans, config.seasons, config.stepDate
    )

    # read climatological EOFs
    eof, clim_ind = get_climatology_eof(
        config.climClusterCentroidsEOF,
        config.climEOFs,
        config.climPCs,
        config.climSdv,
        config.climClusterIndex,
        config.nClusterClim,
        config.monStartDoS,
        config.monEndDoS,
    )

    for scenario, fname in clusterFiles.items():
        # read grib cluster
        cents, ens_numbers, lat, lon = read_grib_cluster(
            fname, config.stepStart, config.stepDel, config.nSteps, config.nEns
        )
        # ? outout ens_numbers?

        # get regions mask
        mask = region_weights(*normalise_angles(config.bbox, positive=True), lat, lon)

        weights = lat_weights(lat, mask)

        # compute anomalies
        anom = cents - clim
        anom = np.clip(anom, -config.anMax, config.anMax)
        
        cluster_att, min_dist = attribution(anom, eof, clim_ind, weights)

        # write report output
        # table: attribution cluster index all fc clusters, step 
        np.savetxt(
            pjoin(config.output_root, f'{config.stepStart}_{config.stepEnd}dist_index_{scenario}.txt'), min_dist,
            fmt='%-10.5f', delimiter=3*' '
        )

        # table: distance measure for all fc clusters, step
        np.savetxt(
            pjoin(config.output_root, f'{config.stepStart}_{config.stepEnd}att_index_{scenario}.txt'), cluster_att,
            fmt='%-3d', delimiter=3*' '
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
