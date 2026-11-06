# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import abc
import json

import numpy as np
import pandas as pd
import xarray as xr
from earthkit.meteo.regimes import (
    ConstantPatterns,
    ModulatedPatterns,
    project,
    regime_index,
)


def needs_valid_time(f):
    """Allow functions requiring valid time to also accept forecast_reference_time and step"""

    def wrapped(*args, valid_time=None, **kwargs):
        if valid_time is None:
            forecast_reference_time = kwargs.pop("forecast_reference_time", None)
            step = kwargs.pop("step", None)
            assert forecast_reference_time is not None and step is not None
            valid_time = np.asarray(forecast_reference_time) + np.asarray(step)
        return f(*args, valid_time=valid_time, **kwargs)

    return wrapped


def detect_time_coords(anomaly: xr.DataArray) -> list[str]:
    """Detect the time coordinates for use as patterns_coords"""
    if "valid_time" in anomaly.dims:
        return ["valid_time"]
    forecast_dims = ["forecast_reference_time", "step"]
    if all(dim in anomaly.dims for dim in forecast_dims):
        return forecast_dims
    raise ValueError(
        "Cannot determine time coordinates: expected a 'valid_time' dimension or "
        f"'forecast_reference_time' and 'step' dimensions, got {list(anomaly.dims)}"
    )


class RegimesClassifier(abc.ABC):
    def __init__(self, path):
        self.path = path
        self.data = xr.load_dataset(self.path)

    @abc.abstractmethod
    def classify(self):
        pass

    def crop_region(self, ds):
        return ds.sel(
            {
                "latitude": self.data.coords["latitude"].values,
                "longitude": self.data.coords["longitude"].values,
            }
        )


class NAOBLRegimesClassifier(RegimesClassifier):
    def __init__(self, path):
        super().__init__(path)
        self.patterns = ConstantPatterns(
            labels=self.data.coords["pattern"].values.tolist(),
            patterns=self.data["eof_pattern"].values,
            grid=json.loads(self.data.attrs["earthkit_grid_spec"]),
        )

    @property
    def projection_mean(self):
        return self.data["projection_mean"]

    @property
    def projection_std(self):
        return self.data["projection_std"]

    def classify(self, anomaly, **kwargs):
        kwargs = {"patterns_coords": None, **kwargs}
        weights = self.data["gridpoint_weights"]
        proj = project(anomaly, self.patterns, weights, **kwargs)
        index = regime_index(proj, self.projection_mean, self.projection_std)
        # TODO clear/set attributes
        return index


class Grams7RegimeClassifier(RegimesClassifier):
    def __init__(self, path):
        self.path = path
        self.data = xr.load_dataset(path)
        modulator_dims = ["month", "day", "hour"]
        self.modulator_series = self.data["pattern_modulator"].to_series()
        self.modulator_series.index = pd.MultiIndex.from_arrays(
            [self.data.coords[dim].values for dim in modulator_dims],
            names=modulator_dims,
        )
        self.patterns = ModulatedPatterns(
            labels=self.data.coords["pattern"].values.tolist(),
            base_patterns=self.data["base_pattern"].values,
            modulator=self.pattern_modulator,
            grid=json.loads(self.data["base_pattern"].attrs["earthkit_grid_spec"]),
        )

    @property
    def projection_mean(self):
        return self.data["projection_mean"]

    @property
    def projection_std(self):
        return self.data["projection_std"]

    @needs_valid_time
    def pattern_modulator(self, *, valid_time):
        shape = valid_time.shape
        dates = pd.to_datetime(valid_time.ravel())
        return self.modulator_series.loc[
            list(zip(dates.month, dates.day, dates.hour))
        ].values.reshape(shape)

    def classify(self, anomaly, **kwargs):
        kwargs = {"patterns_coords": detect_time_coords(anomaly), **kwargs}
        weights = np.cos(np.deg2rad(self.data.coords["latitude"]))
        proj = project(anomaly, self.patterns, weights, **kwargs)
        index = regime_index(proj, self.projection_mean, self.projection_std)
        # TODO: clear/set attributes
        return index
