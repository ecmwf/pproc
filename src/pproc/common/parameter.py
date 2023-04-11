import datetime
from typing import Dict, List
import numpy as np
import numexpr

from pproc import common


def create_parameter(
    date: datetime.datetime, global_input_cfg: Dict, param_cfg: Dict, n_ensembles: int
):
    if "input_combine_operation" in param_cfg:
        param_ids = param_cfg["in_paramid"].split("/")
        assert len(param_ids) == 2
        return CombineParameters(
            date, param_ids, global_input_cfg, param_cfg, n_ensembles
        )
    if "input_filter_operation" in param_cfg:
        return FilterParameter(
            date, param_cfg["in_paramid"], global_input_cfg, param_cfg, n_ensembles
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
        self.scale_data = int(param_cfg.get("scale", 1))

    def retrieve_data(self, fdb, step: int):
        combined_data = []
        for type in self.base_request["type"].split("/"):
            new_request = self.base_request.copy()
            new_request["step"] = step
            new_request["type"] = type
            if type == "cf":
                new_request.pop("number")
            print("FDB request: ", new_request)
            message_temp, new_data = common.fdb_read_with_template(
                fdb, new_request, self.interpolation_keys
            )
            if len(combined_data) == 0:
                combined_data = new_data
            else:
                combined_data = np.concatenate((combined_data, new_data), axis=0)

        return message_temp, combined_data * self.scale_data

    def type_and_number(self, index: int):
        """
        Get data type and ensemble number from concatenated data index
        """
        types = self.base_request["type"].split("/")
        if "pf" in types:
            nensembles = len(self.base_request["number"])
            pf_start_index = types.index("pf")
            if index < pf_start_index:
                return types[index], 0
            if index < pf_start_index + nensembles:
                return "pf", index - pf_start_index + 1
            return types[index - (nensembles - 1)], 0
        else:
            return types[index], 0

    def get_type_index(self, type: str):
        """
        Get range of concatenated data indices for requested type
        """
        types = self.base_request["type"].split("/")
        index = types.index(type)
        nensembles = len(self.base_request["number"])
        if type == "pf":
            pf_start_index = types.index("pf")
            return range(pf_start_index, pf_start_index + nensembles)
        if "pf" in types:
            pf_start_index = types.index("pf")
            if index > pf_start_index:
                offset = pf_start_index + nensembles - 1
                return range(offset + index, offset + index + 1)
        return range(index, index + 1)


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


class FilterParameter(Parameter):
    """
    Class for digesting parameter related config and retrieving parameter data
    Filters input data based on its own values, or the values of another parameter
    Filtering is specified by:
        - comparison: operation to compare values
        - threshold: value to compare data against
        - param: parameter data used in filter, input data itself is used if none is specified
        - replacement: value to replace all data values by that satisfy the filter, default is 0
    """

    def __init__(
        self,
        dt: datetime.datetime,
        param_id: int,
        global_input_cfg: Dict,
        param_cfg: Dict,
        n_ensembles: int,
    ):
        super().__init__(dt, param_id, global_input_cfg, param_cfg, n_ensembles)
        self.param_id = param_id
        self.filter_comparison = param_cfg["input_filter_operation"]["comparison"]
        self.filter_threshold = param_cfg["input_filter_operation"]["threshold"]
        self.filter_param = param_cfg["input_filter_operation"].get("param", param_id)
        self.filter_replacement = float(
            param_cfg["input_filter_operation"].get("replacement", 0)
        )

    def retrieve_data(self, fdb, step: int):
        msg_template, data = super().retrieve_data(fdb, step)

        filter_data = data
        if self.filter_param != self.param_id:
            self.base_request["param"] = self.filter_param
            _, filter_data = super().retrieve_data(fdb, step)
            # Reset back to original
            self.base_request["param"] = self.param_id

        comp = numexpr.evaluate(
            "data " + self.filter_comparison + str(self.filter_threshold),
            local_dict={"data": filter_data},
        )
        return msg_template, np.where(comp, self.filter_replacement, data)
