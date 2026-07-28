import os
import pytest 
from typing import Optional
import numpy as np

import earthkit.data 

from ppruntime import thermal_indices

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(TEST_DIR, "data")


@pytest.mark.parametrize(
    "function", [
        thermal_indices.calc_mrt,
        thermal_indices.calc_aptmp,
        thermal_indices.calc_wcf,
        thermal_indices.calc_nefft,
        thermal_indices.calc_wbt,
        thermal_indices.calc_gt, 
        thermal_indices.calc_wbgt,
        thermal_indices.calc_utci,
        thermal_indices.calc_dsrp,
        thermal_indices.calc_heatx,
        thermal_indices.calc_rhp,
        thermal_indices.calc_hmdx,
        thermal_indices.calc_cossza,
    ]
)
def test_thermal_indices(monkeypatch, function):
    inputs: earthkit.data.FieldList = earthkit.data.from_source("file", os.path.join(DATA_DIR, "test_2t_12.grib")) # type: ignore
    def patched_sel(paramId: Optional[int] = None, stepType: Optional[str] = None):
        if paramId is not None:
            selected = inputs.sel(param="2t")
            if paramId == 214001:
                # To avoid invalid value warnings for arcsin
                return earthkit.data.FieldList.from_array(
                np.random.rand(*selected.values.shape), selected[0].metadata().override(stepType="diff", stepRange="0-12")
            )
            return selected
        if stepType is not None:
            param = inputs.sel(param="2t")
            return earthkit.data.FieldList.from_array(
                param.values, param[0].metadata().override(stepType="diff", stepRange="0-12")
            )
    ds = earthkit.data.FieldList.from_array(inputs.values, inputs.metadata())
    monkeypatch.setattr(ds, "sel", patched_sel)
    function(ds, metadata={"edition": 2})