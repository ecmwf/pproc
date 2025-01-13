import datetime
from typing import Any, Dict, List, Union
import numpy as np
import xarray as xr
import numexpr

from earthkit.meteo.wind import direction

from pproc import common


def create_parameter(
    name: str,
    date: datetime.datetime,
    global_input_cfg: Dict,
    param_cfg: Dict,
    n_ensembles: int,
    overrides: Dict[str, Any] = {},
):
    if "input_combine_operation" in param_cfg:
        param_ids = param_cfg["in_paramid"].split("/")
        return CombineParameters(
            name, date, param_ids, global_input_cfg, param_cfg, n_ensembles, overrides
        )
    if "input_filter_operation" in param_cfg:
        return FilterParameter(
            name,
            date,
            param_cfg["in_paramid"],
            global_input_cfg,
            param_cfg,
            n_ensembles,
            overrides,
        )
    return Parameter(
        name,
        date,
        param_cfg["in_paramid"],
        global_input_cfg,
        param_cfg,
        n_ensembles,
        overrides,
    )


class Parameter:
    """
    Class for digesting parameter related config and retrieving parameter data
    """

    def __init__(
        self,
        name: str,
        dt: datetime.datetime,
        param_id: int,
        global_input_cfg,
        param_cfg: Dict,
        n_ensembles: Union[int, range],
        overrides: Dict[str, Any] = {},
    ):
        self.name = name
        self.base_request = global_input_cfg.copy()
        self.base_request.update(param_cfg["base_request"])
        self.base_request["param"] = param_id
        self.members = n_ensembles
        self.base_request["date"] = dt.strftime("%Y%m%d")
        self.base_request["time"] = dt.strftime("%H")
        self.overrides = overrides
        self.interpolation_keys = param_cfg.get("interpolation_keys", None)
        self.scale_data = int(param_cfg.get("scale", 1))

    def retrieve_data(
        self, step: common.AnyStep, join_dim: str = "number", **kwargs
    ):
        combined_data = []
        for tp in self.base_request["type"].split("/"):
            new_request = self.base_request.copy()
            new_request["step"] = str(step)
            new_request.update(kwargs)
            new_request["type"] = tp
            if tp == "pf":
                new_request["number"] = (
                    range(1, self.members + 1)
                    if isinstance(self.members, int)
                    else self.members
                )
            elif tp == "fcmean":
                new_request["number"] = (
                    range(0, self.members + 1)
                    if isinstance(self.members, int)
                    else self.members
                )
            new_request.update(self.overrides)
            print("FDB request: ", new_request)
            new_data = common.fdb_read(common.io.fdb(), new_request, self.interpolation_keys)

            members = new_request.get("number", [0])
            if tp in ["cf", "fc"]:
                new_data = new_data.expand_dims({"number": members})
            if "number" in new_data.dims:
                assert new_data.sizes["number"] == len(members)
            if num_levels := len(self.levels()) > 1:
                assert new_data.sizes["levelist"] == num_levels

            if join_dim not in new_data.dims:
                new_data = new_data.expand_dims({join_dim: [new_request[join_dim]]})
            if len(combined_data) == 0:
                combined_data = new_data
            else:
                combined_data = xr.concat([combined_data, new_data], dim=join_dim)

        return combined_data.attrs.pop("grib_template"), combined_data * self.scale_data

    def levels(self):
        levelist = self.base_request.get("levelist", [0])
        if isinstance(levelist, int):
            return [levelist]
        return levelist


class CombineParameters(Parameter):
    def __init__(
        self,
        name: str,
        dt: datetime.datetime,
        param_ids: List[int],
        global_input_cfg: Dict,
        param_cfg: Dict,
        n_ensembles: int,
        overrides: Dict[str, Any] = {},
    ):
        super().__init__(
            name, dt, 0, global_input_cfg, param_cfg, n_ensembles, overrides
        )
        self.param_ids = param_ids
        self.combine_operation = param_cfg["input_combine_operation"]

    def combine_data(self, data_list):
        if self.combine_operation == "norm":
            return np.linalg.norm(data_list, axis=0)
        if self.combine_operation == "direction":
            assert len(data_list) == 2, "'direction' requires exactly 2 input fields"
            return direction(
                data_list[0], data_list[1], convention="meteo", to_positive=True
            )
        return getattr(np, self.combine_operation)(data_list, axis=0)

    def retrieve_data(self, step: common.AnyStep, **kwargs):
        data_list = []
        for param_id in self.param_ids:
            self.base_request["param"] = param_id
            msg_template, data = super().retrieve_data(step=step, **kwargs)
            data_list.append(data)

        res = self.combine_data(data_list)
        da_template = data_list[0]
        return msg_template, xr.DataArray(
            res,
            dims=da_template.dims,
            coords=da_template.coords,
            attrs=da_template.attrs,
        )


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
        name: str,
        dt: datetime.datetime,
        param_id: int,
        global_input_cfg: Dict,
        param_cfg: Dict,
        n_ensembles: int,
        overrides: Dict[str, Any] = {},
    ):
        super().__init__(
            name, dt, param_id, global_input_cfg, param_cfg, n_ensembles, overrides
        )
        self.param_id = param_id
        self.filter_comparison = param_cfg["input_filter_operation"]["comparison"]
        self.filter_threshold = param_cfg["input_filter_operation"]["threshold"]
        self.filter_param = param_cfg["input_filter_operation"].get("param", param_id)
        self.filter_replacement = float(
            param_cfg["input_filter_operation"].get("replacement", 0)
        )

    def retrieve_data(self, step: common.AnyStep, **kwargs):
        msg_template, data = super().retrieve_data(step=step, **kwargs)

        filter_data = data
        if self.filter_param != self.param_id:
            self.base_request["param"] = self.filter_param
            _, filter_data = super().retrieve_data(step=step, **kwargs)
            # Reset back to original
            self.base_request["param"] = self.param_id

        comp = numexpr.evaluate(
            "data " + self.filter_comparison + str(self.filter_threshold),
            local_dict={"data": filter_data},
        )
        res = np.where(comp, self.filter_replacement, data)
        return msg_template, xr.DataArray(
            res, dims=data.dims, coords=data.coords, attrs=data.attrs
        )
