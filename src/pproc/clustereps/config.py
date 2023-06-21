
import argparse
from datetime import datetime, timedelta
from os.path import join as pjoin
from typing import List

from pproc.common import Config
from pproc.clustereps.season import MONTH_DAYS, SeasonConfig
from pproc.clustereps.utils import gen_steps


class ClusterConfigBase(Config):
    def __init__(self, args, verbose=True):
        super().__init__(args, verbose=verbose)

        # Step range
        self.step_start = int(self.options['step_start'])
        self.step_end = int(self.options['step_end'])
        self.step_del = int(self.options['step_del'])

        # Bounding box
        self.lat_n = float(self.options['lat_n'])
        self.lat_s = float(self.options['lat_s'])
        self.lon_w = float(self.options['lon_w'])
        self.lon_e = float(self.options['lon_e'])
        self.bbox = (self.lat_n, self.lat_s, self.lon_w, self.lon_e)

        # Number of members (51)
        self.num_members = int(self.options.get('num_members', 51))

        # Maximum absolute value of anomalies
        self.clip = float(self.options.get('max_anom', 10000.))

        self.sources = self.options.get('sources', {})

        self.out_keys = self.options.get('out_keys', {})

    @property
    def monthly(self) -> bool:
        return (self.step_end - self.step_start > 120) or (self.step_end == self.step_start)

    @property
    def steps(self) -> List[int]:
        return gen_steps(self.step_start, self.step_end, self.step_del, monthly=self.monthly)


class PCAConfig(ClusterConfigBase):
    def __init__(self, args, verbose=True):
        super().__init__(args, verbose=verbose)

        # Normalisation factor (1)
        self.pca_factor = self.options.get('pca_factor', None)
        # Number of components to extract
        self.ncomp = int(self.options['num_components'])


class ClusterConfig(ClusterConfigBase):
    def __init__(self, args, verbose=True):

        super().__init__(args, verbose=verbose)

        # Variance threshold
        self.var_th = float(self.options['var_th'])
        # Number of PCs to use, optional
        self.npc = int(self.options.get('npc', -1))
        # Normalisation factor (2/5)
        self.cluster_factor = float(self.options.get('cluster_factor', 0.4))
        # Max number of clusters
        self.ncl_max = int(self.options['ncl_max'])
        # Number of clustering passes
        self.npass = int(self.options['npass'])
        # Number of red-noise samples for significance computation
        self.nrsamples = int(self.options['nrsamples'])
        # Maximum significance threshold
        self.max_sig = float(self.options['max_sig'])
        # Medium significance threshold
        self.med_sig = float(self.options['med_sig'])
        # Minimum significance threshold
        self.min_sig = float(self.options['min_sig'])
        # Significance tolerance
        self.sig_tol = float(self.options['sig_tol'])
        # Parallel red-noise sampling
        self.n_par = int(self.options.get('n_par', 1))

        # Initialisation method (k-means++ or sector)
        self.init = self.options.get('init', "k-means++")
        if self.init not in ["k-means++", "sector"]:
            raise RuntimeError(f"Invalid initialisation method {self.init!r}")
        if verbose:
            print(f"Initialisation method: {self.init}")


class AttributionConfig(ClusterConfigBase):
    def __init__(self, args: argparse.Namespace, verbose: bool = True) -> None:

        super().__init__(args, verbose=verbose)

        # Climatological data options
        self.nClusterClim = int(self.options.get('ncl_clim', 6))
        self.climMeans = pjoin(args.clim_dir, self.options['clim_means'])
        self.climPCs = pjoin(args.clim_dir, self.options['clim_pcs'])
        self.climSdv = pjoin(args.clim_dir, self.options['clim_sdv'])
        self.climEOFs = pjoin(args.clim_dir, self.options['clim_eof'])
        self.climClusterIndex = pjoin(args.clim_dir, self.options['clim_cluster_index'])
        self.climClusterCentroidsEOF = pjoin(args.clim_dir, self.options['clim_cluster_centroids_eof'])
        # pca options
        self._seasons = self.options.get('seasons', [(1, 12)])
        self._seasons = [(int(tup[0]), int(tup[1])) for tup in self._seasons]

        # forecast date
        self.date = args.date
        self.year = self.date.year
        self.month = self.date.month

        self.nSteps = 1 + (self.step_end - self.step_start) // self.step_del
        self.stepDate = self.date + timedelta(hours=self.step_start)

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


    def __repr__(self) -> str:
        return self.__dict__.__repr__()


class FullClusterConfig(AttributionConfig, ClusterConfig, PCAConfig):
    def __init__(self, args: argparse.Namespace, verbose: bool = True):
        super().__init__(args, verbose=verbose)

        self.dummy = self.options.get("generate_dummy", False)
        self.ncl_dummy = self.ncl_max if self.dummy else None