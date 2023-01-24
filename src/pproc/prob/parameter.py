import datetime
from typing import Dict

from pproc import common


class Parameter:
    """
    Class for digesting parameter related config and retrieving parameter data
    """
    def __init__(
        self, dt: datetime.datetime, param_id: int, cfg: Dict, n_ensembles: int
    ):
        self.base_request = cfg["base_request"].copy()
        self.base_request["param"] = param_id
        self.base_request["number"] = range(1, n_ensembles + 1)
        self.base_request["date"] = dt.strftime("%Y%m%d")
        self.base_request["time"] = dt.strftime("%H")
        self.interpolation_keys = cfg.get("interpolation_keys", None)

    def retrieve_data(self, fdb, step: int):
        new_request = self.base_request
        new_request["step"] = step

        return common.fdb_read_with_template(fdb, new_request, self.interpolation_keys)
