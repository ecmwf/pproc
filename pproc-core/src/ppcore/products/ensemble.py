# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Optional, Iterator
from dataclasses import dataclass

from earthkit.workflows.plugins.pproc.fluent import Action
from earthkit.workflows.plugins.pproc.fluent import from_source
from earthkit.workflows.plugins.pproc.utils.request import Request

from ppcore.utils.requests import validate_request
from ppcore.utils.mars import extract_mars
from ppcore.configs.product.ensemble import Config
from ppcore.products.base import Product


@dataclass
class Ensemble(Product):
    config: Config
    preprocessing_dim: str = "param"
    ensemble_dim: str = "number"

    def source(self, ensemble_dim: Optional[str] = None) -> Action:
        if self.config.inputs is None:
            raise ValueError("No inputs provided for ensemble config")
        input_config = self.config.inputs.fc
        ensemble_dim = ensemble_dim or self.ensemble_dim
        requests = []
        for x in [dict(x, **self.input_overrides) for x in input_config.requests]:
            req = Request(validate_request(x), no_expand=("levelist"))
            if "number" not in req and ensemble_dim == "number":
                req.make_dim("number", 0)
            requests.append(req)
        return from_source(
            input_config.sources,
            requests,
            input_config.dtype,
            join_key=ensemble_dim,
        )

    def action(
        self,
        forecast: Optional[Action] = None,
        preprocessing_dim: Optional[str] = None,
        ensemble_dim: Optional[str] = None,
    ) -> Action:
        ret = forecast or self.source(ensemble_dim=ensemble_dim)
        for preprocessing in self.config.preprocessing.actions:
            ret = ret.preprocessing(
                preprocessing_dim or self.preprocessing_dim,
                **preprocessing.model_dump(),
            )
        for dim, accumulation in self.config.accumulations.items():
            ret = ret.accumulation(
                dim=dim,
                **accumulation.model_dump(),
            )
        ret = ret.ensemble_statistics(
            dim=ensemble_dim or self.ensemble_dim,
            **self.config.statistics.model_dump(),
        )

        if self.config.output is not None:
            output_config = self.config.output.model_dump()
            out_metadata = output_config["metadata"].copy()
            out_metadata.update(self.output_overrides)
            ret = ret.write(output_config["targets"], metadata=out_metadata)
        return ret

    def in_mars(self, sources: Optional[list[str]] = None) -> Iterator[dict]:
        if self.config.inputs is None:
            return
        inputs = self.config.inputs.fc
        if not sources or set.intersection(
            set(sources), set(x.name for x in inputs.sources)
        ):
            source = inputs.sources[0]
            for input in inputs.requests:
                overridden = input.copy()
                overridden = extract_mars(overridden)
                if source.name == "file":
                    overridden["source"] = source.path
                elif source.name == "file-pattern":
                    overridden["source"] = source.pattern
                else:
                    overridden["source"] = source.name
                overridden.update(self.input_overrides)
                yield overridden

    def out_mars(self, targets: Optional[list[str]] = None) -> Iterator[dict]:
        output = self.config.output
        if output is None:
            return
        if not targets or set.intersection(
            set(targets), set(x.name for x in output.targets)
        ):
            target = output.targets[0]
            overridden = output.request.copy()
            overridden = extract_mars(overridden)
            if target.name == "file":
                overridden["target"] = target.file
            elif target.name == "file-pattern":
                overridden["target"] = target.file
            else:
                overridden["target"] = target.name
            overridden.update(self.output_overrides)
            yield overridden
