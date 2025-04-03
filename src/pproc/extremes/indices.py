from abc import ABCMeta, abstractmethod
from typing import Any, Container, Dict

from earthkit.meteo import extreme
from eccodes import GRIBMessage
import numpy as np

from pproc import common
from pproc.common.io import Target
from pproc.extremes.grib import (
    cpf_template,
    efi_template,
    efi_template_control,
    sot_template,
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
    ):
        raise NotImplementedError


class EFI(Index):
    def __init__(self, options):
        super().__init__(options)
        self.eps = float(options["eps"]) if "eps" in options else -1.0

    def compute(
        self,
        clim: np.ndarray,
        ens: np.ndarray,
        target: Target,
        in_template: GRIBMessage,
        out_template: GRIBMessage,
    ):
        if in_template.get("type") in ["cf", "fc"]:
            efi_control = extreme.efi(clim, ens[:1, :], self.eps)
            template_efi = efi_template_control(out_template)
            common.write_grib(target, template_efi, efi_control)

        efi = extreme.efi(clim, ens, self.eps)
        template_efi = efi_template(out_template)
        common.write_grib(target, template_efi, efi)


class SOT(Index):
    def __init__(self, options):
        super().__init__(options)
        self.eps = float(options["eps"]) if "eps" in options else -1.0
        self.sot = list(map(int, options["sot"])) if "sot" in options else []

    def compute(
        self,
        clim: np.ndarray,
        ens: np.ndarray,
        target: Target,
        in_template: GRIBMessage,
        out_template: GRIBMessage,
    ):
        for perc in self.sot:
            sot = extreme.sot(clim, ens, perc, self.eps)
            template_sot = sot_template(out_template, perc)
            common.write_grib(target, template_sot, sot)


class CPF(Index):
    def __init__(self, options):
        super().__init__(options)
        self.eps = float(options["cpf_eps"]) if "cpf_eps" in options else None
        self.symmetric = options.get("cpf_symmetric", False)

    def compute(
        self,
        clim: np.ndarray,
        ens: np.ndarray,
        target: Target,
        in_template: GRIBMessage,
        out_template: GRIBMessage,
    ):
        cpf = extreme.cpf(
            clim.astype(np.float32),
            ens.astype(np.float32),
            sort_clim=False,
            sort_ens=True,
            epsilon=self.eps,
            symmetric=self.symmetric,
        )
        template_cpf = cpf_template(out_template)
        common.write_grib(target, template_cpf, cpf)


_INDICES = {"efi": EFI, "sot": SOT, "cpf": CPF}
SUPPORTED_INDICES = ["efi", "sot", "cpf"]


def create_indices(
    options: Dict[str, Any], default_indices: Container[str] = ()
) -> Dict[str, Index]:
    indices = {}
    for index, index_cls in _INDICES.items():
        enabled = options.get(f"compute_{index}", index in default_indices)
        if enabled:
            indices[index] = index_cls(options)
    return indices
