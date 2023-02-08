import datetime
from typing import Dict, List
import numpy as np

from pproc import common


def create_parameter(date: datetime.datetime, cfg: Dict, n_ensembles: int):
    if isinstance(cfg["in_paramid"], str) and "/" in cfg["in_paramid"]:
        param_ids = cfg["in_paramid"].split("/")
        assert len(param_ids) == 2
        return CombineParameters(date, param_ids, cfg, n_ensembles)
    return Parameter(date, cfg["in_paramid"], cfg, n_ensembles)


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


class CombineParameters(Parameter):
    def __init__(
        self, dt: datetime.datetime, param_ids: List[int], cfg: Dict, n_ensembles: int
    ):
        super().__init__(dt, 0, cfg, n_ensembles)
        self.param_ids = param_ids
        self.combine_operation = cfg["input_combine_operation"]

    def combine_data(self, data_list):
        if self.combine_operation == "norm":
            return np.linalg.norm(data_list, axis=0)
        return getattr(np, self.combine_operation)(data_list, axis=0)

    def retrieve_data(self, fdb, step: int):
        new_request = self.base_request
        new_request["step"] = step

        data_list = []
        for param_id in self.param_ids:
            new_request["param"] = param_id
            msg_template, data = common.fdb_read_with_template(
                fdb, new_request, self.interpolation_keys
            )
            data_list.append(data)

        return msg_template, self.combine_data(data_list)
