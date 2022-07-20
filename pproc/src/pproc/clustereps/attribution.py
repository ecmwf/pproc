import sys
import argparse
from datetime import datetime, timedelta
from calendar import isleap
from typing import List, Tuple
import os

import yaml
import numpy as np
from scipy.io import FortranFile

from eccodes import FileReader

from pproc.clustereps.utils import region_weights, lat_weights, normalise_angles


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

    def __repr__(self) -> str:
        return f"Season ({self.name}) - {self.start:%d/%m/%Y}: {self.end:%d/%m/%Y} ({self.ndays} days)"


class AttributionConfig:
    def __init__(self, options: dict, date: datetime, output_root: str) -> None:

        # yaml config file
        self.stepStart = options.get('stepStart', 0)
        self.stepEnd = options.get('stepEnd', 120)
        self.stepDel = options.get('stepDel', 12)
        # main grid specs
        self.nLat = options['nLat']
        self.nLon = options['nLon']
        # EOF grid dimensions
        self.rLatN = options.get('rLatN', 90.)
        self.rLatS = options.get('rLatS', -90.)
        self.rLonW = options.get('rLonW', 0.)
        self.rLonE = options.get('rLonE', 360.)
        # forecast options
        self.nEns = options.get('nEns', 51)
        # Climatological data options
        self.nClusterClim = options.get('nClusterClim', 6)
        self.climPCs = options['climPCs']
        self.climSdv = options['climSdv']
        self.climEOFs = options['climEOFs']
        self.climClusterIndex = options['climClusterIndex']
        self.climClusterCentroidsEOF = options['climClusterCentroidsEOF']
        # pca options
        self.anMax = options.get('anMax', 10000.)
        self._seasons = options.get('seasons', [(1, 12)])
        
        # forecast date
        self.date = date
        self.year = self.date.year
        self.month = self.date.month
        self.fcDoy = self.date.timetuple().tm_yday

        # adjust indexes to ignore leap years
        if isleap(self.date.year) and self.dateDoy > 59:
            self.fcDoy -= 1
            self.monEndDoy -= 1
            self.monStartDoy -= 1

        self.stepDay = self.stepStart // 24
        self.nSteps = 1 + (self.stepEnd - self.stepStart) // self.stepDel
        refDayIndex = self.fcDoy + self.stepDay - 1
        if refDayIndex > 364:
            refDayIndex = refDayIndex - 364
        self.stepDoy = refDayIndex

        # out directory
        self.output_root =  output_root

        # seasons parsing
        _seasons = []
        for startMonth, endMonth in self._seasons:
            _seasons.append(Season(startMonth, endMonth, self.year))
        self.seasons = _seasons

        for sea in self.seasons:
            if self.fcDoy in sea.doys:
                self.thisSeason = sea
                break
        self.monStartDoy = self.thisSeason.dos(datetime(self.year, self.month, 1))
        self.monEndDoy = self.thisSeason.dos(datetime(self.year, self.month, MONTH_DAYS[self.month - 1]))
        
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

    parser = argparse.ArgumentParser(description='Cluster attribution to closest partition of climatological weather regimes')
    parser.add_argument('config', type=argparse.FileType('r'), help="Configuration file (YAML)")
    parser.add_argument('date', help='forecast date (YYMMDD)', type=lambda x: datetime.strptime(x, '%Y%m%d'))
    parser.add_argument('centroids', help='Forecast cluster centroids (NPZ)')
    parser.add_argument('representative', help='Forecast cluster representative (NPZ)')
    parser.add_argument('-o', '--output_root', default=os.getcwd(), help='output base directory')

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
                cents = np.empty((nclusters, nSteps, npoints)).astype('float64')
                ens_numbers = np.empty((nclusters, nEns)).astype('int16')
                lat = message.get('latitudes')
                lon = message.get('longitudes')
            icl = message.get('clusterNumber') - 1
            jstep = (message.get('step') - stepStart) // stepDelta
            # get field values
            cents[icl, jstep, :] = message.get_array('values')
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


def get_climatology_fields(nPoints: int, seasons: List[Season], dayIndex: int):
    """_summary_

    Parameters
    ----------
    nPoints : int
        total number of grid points
    seasons : List[Season]
        list of season definitions

    Returns
    -------
    np.Array (nPoints)
        field array values from GRIB
    """
    # read all seasons and merge to one year climatology file (365, npoints)
    fname_template = '{season}_means.grd'
    year_clim = np.empty((365, nPoints))
    for sea in seasons:
        sea_clim = read_clim_file(fname_template.format(season=sea.name), sea.ndays)
        year_clim[sea.doys, :] = sea_clim
    
    return year_clim[dayIndex, :]


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
        Start of month in day of year.
    monEnd : int
        End of month in day of year.

    Returns
    -------
    np.Array (nEOF, nPoints)
        Climatological EOFs
    np.ndarray (nClusterClim, nEOF)
        Mean monthly principal components for each cluster and EOF
    """

    grid_specs_file = climClusterInfo
    neof, nyrs, ndays = read_clim_file(grid_specs_file, 1, ">u4")

    #assert ndays == season.ndays, "climatology file must match season definition"

    eof = read_clim_file(climEOFs, neof)

    pcs = read_clim_file(climPCs, nyrs*ndays)
    pcs = pcs[:, :neof]
    pcs = pcs.reshape(ndays, nyrs, neof)

    sdv = read_clim_file(climSdv, 1)[:neof]

    # Compute non-standardized PCs and PCs total variance
    pcs = pcs * sdv
    pcs = np.moveaxis(pcs, -1, 0)

    clus_index = read_clim_file(climIndex, ndays*nyrs)
    clus_index = clus_index[:, nClusters-1]
    clus_index = clus_index.reshape(ndays, nyrs)

    # compute cluster centroids in EOF space for the corresponding month.
    # ? output?
    month_ind = (clus_index[monStart: monEnd + 1, :]).flatten().astype(np.int16)
    clcases = np.bincount(
        month_ind,
        weights=np.ones_like(month_ind).astype(np.int8),
        minlength=nClusters
    )
    clcases = clcases[1:]  # drop cluster 0

    monDays = monEnd - monStart + 1
    mon_pcs = pcs[:, monStart: monEnd + 1, :]
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

    # ? Check on clim eof grid? in fortran there is a "change_grid" routine but it's never triggered and ngpeof is forcelly set to ngp in frame_attribute executable
    # 3.3) project anomalies onto climatological eof
    norm = np.sqrt(np.einsum('ij,ij,j->i', climEOF, climEOF, weights))
    anom_proj = np.einsum('zijk,jk,k->zij', fcField[:, :, np.newaxis, :], climEOF, weights) / norm

    #del eof

    # 3.4) Compute distance between clusters and weather regimes
    diff = anom_proj[:, :, np.newaxis, :] - climIndex
    rms = np.sqrt((diff ** 2).mean(axis=-1))
    min_dist = rms.min(axis=-1)
    cluster_index = rms.argmin(axis=-1) + 1

    return cluster_index, min_dist


def main(config: AttributionConfig, clusterFiles: dict) -> int:

    # read climatology fields
    clim = get_climatology_fields(config.nPoints, config.seasons, config.stepDoy)

    # read climatological EOFs
    eof, clim_ind = get_climatology_eof(
        config.climClusterCentroidsEOF,
        config.climEOFs,
        config.climPCs,
        config.climSdv,
        config.climClusterIndex,
        config.nClusterClim,
        config.monStartDoy,
        config.monEndDoy,
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
            os.path.join(config.output_root, f'{config.stepStart}_{config.stepEnd}dist_index_{scenario}.txt'), min_dist,
            fmt='%-10.5f', delimiter=3*' '
        )

        # table: distance measure for all fc clusters, step
        np.savetxt(
            os.path.join(config.output_root, f'{config.stepStart}_{config.stepEnd}att_index_{scenario}.txt'), cluster_att,
            fmt='%-3d', delimiter=3*' '
        )
    
    return 0


if __name__ == "__main__":
    parser = get_parser()

    args = parser.parse_args(sys.argv[1:])

    options = yaml.load(args.config, Loader=yaml.SafeLoader)

    config = AttributionConfig(options, args.date, args.output_root)

    to_process = {
        'centroids': args.centroids,
        'representative': args.representative,
    }

    sys.exit(main(config, to_process))
