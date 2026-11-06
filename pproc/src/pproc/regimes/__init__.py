# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pproc.regimes.classifiers import (
    RegimesClassifier,
    NAOBLRegimesClassifier,
    Grams7RegimesClassifier,
)


def get_classifier(name: Literal["grams", "naobl"], path: str) -> RegimesClassifier:
    if name == "grams":
        return Grams7RegimesClassifier(path)
    elif name == "naobl":
        return NAOBLRegimesClassifier(path)
    raise ValueError(name)
