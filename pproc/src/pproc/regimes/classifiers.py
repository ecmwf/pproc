# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import abc
import json
from collections.abc import Sequence

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

    def regime_index(self, proj: xr.DataArray) -> xr.Dataset:
        """Standardise the projections and set the attributes of the index.

        The index is returned as a dataset with the DOI of the pattern data, if
        available, as the global ``references`` attribute.
        """
        index = regime_index(
            proj, self.data["projection_mean"], self.data["projection_std"]
        )
        index.attrs = {
            "long_name": "Regime index",
            "units": "1",
            "description": (
                "Area-weighted regime projection standardised with the "
                "reference mean and standard deviation"
            ),
        }
        ds = index.to_dataset()
        if doi := self.data.attrs.get("original_source_doi"):
            ds.attrs["references"] = f"https://doi.org/{doi}"
        return ds

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

    def classify(self, anomaly, **kwargs):
        kwargs = {"patterns_coords": None, **kwargs}
        weights = self.data["gridpoint_weights"]
        proj = project(anomaly, self.patterns, weights, **kwargs)
        return self.regime_index(proj)


class Grams7RegimesClassifier(RegimesClassifier):
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
        return self.regime_index(proj)

    def filter_regime_index(
        self,
        iwr: xr.DataArray,
        window: Sequence[float],
    ) -> xr.DataArray:
        """Filter a regime index as if the projections had been filtered.

        The Grams scheme is implemented as time-modulated patterns. Filtering
        the index directly mixes different modulations. The filter is applied
        to the unmodulated, unstandardised projections instead, and the
        modulation of the target time is reapplied afterwards:

            I_filtered(t) = a(t) * F[(I + c) / a](t) - c,
            where c = mean / std, F is the filter

        This equals the index of the projection of the filtered anomalies.
        """
        kernel = np.asarray(window, dtype=float)
        if kernel.ndim != 1 or kernel.size % 2 == 0:
            raise ValueError("Filter window must be a 1D kernel of odd length")
        if not np.isclose(kernel.sum(), 1.0):
            raise ValueError(f"Filter window must sum to 1, got {kernel.sum()}")

        valid_time = iwr.coords["valid_time"]
        modulation = xr.DataArray(
            self.pattern_modulator(valid_time=valid_time.values),
            dims=valid_time.dims,
            coords=valid_time.coords,
        )

        c = self.data["projection_mean"] / self.data["projection_std"]
        # Convolution weights apply to y(t - k), the rolling window is in time order
        weights = xr.DataArray(kernel[::-1], dims="__window__")
        shape_proj = (iwr + c) / modulation
        filtered = (
            shape_proj.rolling(valid_time=kernel.size, center=True)
            .construct("__window__")
            .dot(weights)
        )
        result = modulation * filtered - c
        result.name = iwr.name
        result.attrs = iwr.attrs
        return result
