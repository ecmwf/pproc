# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from earthkit.workflows.plugins.pproc.fluent import Action

from .ensemble import EnsembleConfig


class ExtremeConfig(EnsembleConfig):
    @property
    def climatology(self) -> list[dict]:
        return [x for x in self.inputs if x["type"] == "cd"]

    @property
    def forecast(self) -> list[dict]:
        return [x for x in self.inputs if x["type"] != "cd"]

    @property
    def step_range(self) -> str:
        step_accum = self.accumulations["step"]
        if len(step_accum["coords"]) == 1:
            return f"{step_accum['coords'][0]}"
        return f"{step_accum['coords'][0]}-{step_accum['coords'][-1]}"

    def action(
        self,
        forecast: Action,
        climatology: Action,
        preprocessing_dim="param",
        ensemble_dim="number",
    ) -> Action:
        action = forecast
        for preprocessing in self.preprocessing.actions:
            action = action.param_operation(
                dim=preprocessing_dim, **preprocessing.model_dump()
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
        return action.ensemble_extreme(
            climatology=climatology,
            ensemble_dim=ensemble_dim,
            step_ranges=[self.step_range],
            **self.stats,
        )
