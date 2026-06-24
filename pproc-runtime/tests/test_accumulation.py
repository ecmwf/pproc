import os
import numpy as np

import earthkit.data
from ppruntime.accumulation import difference_rate

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(TEST_DIR, "data")


def test_difference_rate():
    fl = earthkit.data.from_source("file", os.path.join(DATA_DIR, "test_2t_12.grib"))
    fl2 = earthkit.data.FieldList.from_array(
        fl.values * 2, [x.override({"step": 24}) for x in fl.metadata()]
    )
    single = difference_rate(fl, factor=0.5)
    assert np.all(single.values == fl.values / 6)
    double = difference_rate(
        fl, fl2, factor=0.5, metadata={"stepType": "diff", "stepRange": "12-24"}
    )
    assert np.all(double.values == fl.values / 6)
    assert double.metadata("stepRange") == ["12-24"]
