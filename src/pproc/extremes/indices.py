# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABCMeta, abstractmethod
from typing import Any, Dict

from earthkit.meteo import extreme
from eccodes import GRIBMessage
import numpy as np

from pproc import common
from pproc.config.targets import Target
from pproc.extremes.grib import (
    cpf_metadata,
    efi_metadata,
    efi_metadata_control,
    sot_metadata,
)


class Index(metaclass=ABCMeta):
    def __init__(self, options):
        pass

    @abstractmethod
    def compute(
        self,
        clim: np.ndarray,
        ens: np.ndarray,
        target: Target,
        in_template: GRIBMessage,
        out_template: GRIBMessage,
        metadata: dict,
    ):
        raise NotImplementedError


class EFI(Index):
    def __init__(self, options):
        super().__init__(options)
        self.eps = float(options.get("eps", -1.0))

    def compute(
        self,
        clim: np.ndarray,
        ens: np.ndarray,
        target: Target,
        in_template: GRIBMessage,
        out_template: GRIBMessage,
        metadata: dict,
    ):
        if in_template.get("type") in ["cf", "fc"]:
            efi_control = extreme.efi(clim, ens[:1, :], self.eps)
            control_keys = efi_metadata_control(out_template, metadata)
            common.io.write_grib(target, out_template, efi_control, control_keys)

        efi = extreme.efi(clim, ens, self.eps)
        efi_keys = efi_metadata(out_template, metadata)
        common.io.write_grib(target, out_template, efi, efi_keys)


class SOT(Index):
    def __init__(self, options):
        super().__init__(options)
        self.eps = float(options.get("eps", -1.0))
        self.sot = options.get("sot", [])

    def compute(
        self,
        clim: np.ndarray,
        ens: np.ndarray,
        target: Target,
        in_template: GRIBMessage,
        out_template: GRIBMessage,
        metadata: dict,
    ):
        for perc in self.sot:
            if isinstance(perc, str) and perc.endswith(":100"):
                # SOT values in GRIB2 are encoded as quantiles 1-10:100, 90-99:100
                percentiles, _ = perc.split(":")
                percentiles = list(map(int, percentiles.split("-")))
                if all(p < 50 for p in percentiles):
                    perc = percentiles[-1]
                elif all(p > 50 for p in percentiles):
                    perc = percentiles[0]
                else:
                    raise ValueError(
                        f"Could not determine SOT percentile value from {perc}"
                    )
            else:
                perc = int(perc)
            sot = extreme.sot(clim, ens, perc, self.eps)
            sot_keys = sot_metadata(out_template, perc, metadata)
            common.io.write_grib(target, out_template, sot, sot_keys)


class CPF(Index):
    def __init__(self, options):
        super().__init__(options)
        self.eps = (
            float(options["cpf_eps"])
            if options.get("cpf_eps", None) is not None
            else None
        )
        self.symmetric = options.get("cpf_symmetric", False)
        self.from_zero = options.get("cpf_from_zero", True)

    def compute(
        self,
        clim: np.ndarray,
        ens: np.ndarray,
        target: Target,
        in_template: GRIBMessage,
        out_template: GRIBMessage,
        metadata: dict,
    ):
        cpf = extreme.cpf(
            clim.astype(np.float32),
            ens.astype(np.float32),
            sort_clim=False,
            sort_ens=True,
            epsilon=self.eps,
            symmetric=self.symmetric,
            from_zero=self.from_zero,
        )
        cpf_keys = cpf_metadata(out_template, metadata)
        common.io.write_grib(target, out_template, cpf, cpf_keys)


_INDICES = {"efi": EFI, "sot": SOT, "cpf": CPF}
SUPPORTED_INDICES = ["efi", "sot", "cpf"]


def create_indices(
    compute_indices: list[str], options: Dict[str, Any]
) -> Dict[str, Index]:
    indices = {}
    for index in compute_indices:
        indices[index] = _INDICES[index](options)
    return indices
