import os
import pytest
from typing import Optional
import numpy as np

import earthkit.data

from ppruntime import thermal_indices

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(TEST_DIR, "data")


@pytest.mark.parametrize(
    "function, paramid",
    [
        [thermal_indices.calc_mrt, 261002],
        [thermal_indices.calc_aptmp, 260255],
        [thermal_indices.calc_wcf, 260005],
        [thermal_indices.calc_nefft, 261018],
        [thermal_indices.calc_wbt, 261023],
        [thermal_indices.calc_gt, 261015],
        [thermal_indices.calc_wbgt, 261014],
        [thermal_indices.calc_utci, 261001],
        [thermal_indices.calc_dsrp, 47],
        [thermal_indices.calc_heatx, 260004],
        [thermal_indices.calc_rhp, 260242],
        [thermal_indices.calc_hmdx, 261016],
        [thermal_indices.calc_cossza, 214001],
    ],
)
def test_thermal_indices(monkeypatch, function, paramid):
    inputs: earthkit.data.FieldList = earthkit.data.from_source(
        "file", os.path.join(DATA_DIR, "test_2t_12.grib")
    )  # type: ignore

    def patched_sel(paramId: Optional[int] = None, stepType: Optional[str] = None):
        if paramId is not None:
            selected = inputs.sel(param="2t")
            if paramId == 214001:
                # To avoid invalid value warnings for arcsin
                return earthkit.data.FieldList.from_array(
                    np.random.rand(*selected.values.shape),
                    selected[0].metadata().override(stepType="diff", stepRange="0-12"),
                )
            return selected
        if stepType is not None:
            param = inputs.sel(param="2t")
            return earthkit.data.FieldList.from_array(
                param.values,
                param[0].metadata().override(stepType="diff", stepRange="0-12"),
            )

    ds = earthkit.data.FieldList.from_array(inputs.values, inputs.metadata())
    monkeypatch.setattr(ds, "sel", patched_sel)
    out = function(ds, metadata={"edition": 2, "paramId": paramid})
    metadata = out[0].metadata()
    assert metadata.get("paramId") == paramid
    assert metadata.get("typeOfFirstFixedSurface") == "sfc"
