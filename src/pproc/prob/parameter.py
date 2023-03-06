import datetime
from typing import Dict, List
import numpy as np

from pproc import common


def create_parameter(
    date: datetime.datetime, global_input_cfg: Dict, param_cfg: Dict, n_ensembles: int
):
    if isinstance(param_cfg["in_paramid"], str) and "/" in param_cfg["in_paramid"]:
        param_ids = param_cfg["in_paramid"].split("/")
        assert len(param_ids) == 2
        return CombineParameters(
            date, param_ids, global_input_cfg, param_cfg, n_ensembles
        )
    return Parameter(
        date, param_cfg["in_paramid"], global_input_cfg, param_cfg, n_ensembles
    )


class Parameter:
    """
    Class for digesting parameter related config and retrieving parameter data
    """

    def __init__(
        self,
        dt: datetime.datetime,
        param_id: int,
        global_input_cfg,
        param_cfg: Dict,
        n_ensembles: int,
    ):
        self.base_request = global_input_cfg.copy()
        self.base_request.update(param_cfg["base_request"])
        self.base_request["param"] = param_id
        self.base_request["number"] = range(1, n_ensembles + 1)
        self.base_request["date"] = dt.strftime("%Y%m%d")
        self.base_request["time"] = dt.strftime("%H")
        self.interpolation_keys = param_cfg.get("interpolation_keys", None)

    def retrieve_data(self, fdb, step: int):
        combined_data = []
        for type in self.base_request['type'].split('/'):
            new_request = self.base_request.copy()
            new_request["step"] = step
            new_request["type"] = type
            if type == 'cf':
                new_request.pop("number")
            message_temp, new_data = common.fdb_read_with_template(
                fdb, new_request, self.interpolation_keys
            )
            if len(combined_data) == 0:
                combined_data = new_data
            else:
                combined_data = np.concatenate((combined_data, new_data), axis=0)

        return message_temp, combined_data


class CombineParameters(Parameter):
    def __init__(
        self,
        dt: datetime.datetime,
        param_ids: List[int],
        global_input_cfg: Dict,
        param_cfg: Dict,
        n_ensembles: int,
    ):
        super().__init__(dt, 0, global_input_cfg, param_cfg, n_ensembles)
        self.param_ids = param_ids
        self.combine_operation = param_cfg["input_combine_operation"]

    def combine_data(self, data_list):
        if self.combine_operation == "norm":
            return np.linalg.norm(data_list, axis=0)
        return getattr(np, self.combine_operation)(data_list, axis=0)

    def retrieve_data(self, fdb, step: int):
        data_list = []
        for param_id in self.param_ids:
            self.base_request["param"] = param_id
            msg_template, data = super().retrieve_data(fdb, step)
            data_list.append(data)

        return msg_template, self.combine_data(data_list)