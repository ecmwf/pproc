# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from earthkit.workflows.backends.earthkit import FieldListBackend
from earthkit.workflows.fluent import Payload

from earthkit.workflows.plugins.pproc.fluent import Action

from .ensemble import EnsembleConfig


class AnomalyConfig(EnsembleConfig):
    @property
    def climatology(self) -> list[dict]:
        return [x for x in self.inputs if x["type"] in ["em", "es"]]

    @property
    def forecast(self) -> list[dict]:
        return [x for x in self.inputs if x["type"] not in ["em", "es"]]

    def action(
        self,
        forecast: Action,
        climatology: Action,
        preprocessing_dim="param",
        ensemble_dim="number",
    ) -> Action:
        clim_headers = climatology.iselect({"type": 0, "step": 0}, drop=True).map(
            Payload(
                "ppruntime.metadata.extract",
                {"keys": ["climateDateFrom", "climateDateTo", "referenceDate"]},
            )
        )
        action = forecast
        for preprocessing in self.preprocessing.actions:
            action = action.param_operation(
                dim=preprocessing_dim, **preprocessing.model_dump()
            )
        action = action.anomaly(
            climatology.select({"type": "em"}, drop=True),
            climatology.select({"type": "es"}, drop=True),
            self.accumulations["step"].get("std_anomaly", False),
        )
        for dim, accumulation in self.accumulations.items():
            action = action.accum_operation(
                accumulation["operation"],
                dim=dim,
                coords=[accumulation["coords"]],
                metadata=accumulation.get("metadata", None),
                include_start=accumulation.get("include_start", False),
                deaccumulate=accumulation.get("deaccumulate", False),
            )
        stats = action.ensemble_operation(
            dim=ensemble_dim,
            **self.stats,
        )
        if self.stats["metadata"].get("edition", 1) == 1:
            return stats
        return stats.join(clim_headers, "**datatype**", match_coord_values=True).reduce(
            FieldListBackend.set_metadata, dim="**datatype**"
        )
