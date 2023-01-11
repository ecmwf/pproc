
from pproc.common import Config


class ClusterConfigBase(Config):
    def __init__(self, args):
        super().__init__(args)

        # Step range
        self.step_start = self.options['step_start']
        self.step_end = self.options['step_end']
        self.step_del = self.options['step_del']

        # Bounding box
        self.lat_n = self.options['lat_n']
        self.lat_s = self.options['lat_s']
        self.lon_w = self.options['lon_w']
        self.lon_e = self.options['lon_e']
        self.bbox = (self.lat_n, self.lat_s, self.lon_w, self.lon_e)

        # Number of members (51)
        self.num_members = self.options.get('num_members', 51)

        # Maximum absolute value of anomalies
        self.clip = self.options.get('max_anom', 10000.)