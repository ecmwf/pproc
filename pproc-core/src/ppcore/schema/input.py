# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import logging
import itertools
from typing import Any
from typing import Iterator
from typing import Optional
from typing import Union
from typing import Literal

from pydantic import field_validator

from qubed import Qube
from earthkit.workflows.plugins.pproc.utils.pydantic_utils import PProcBaseModel
from earthkit.workflows.plugins.pproc.utils.metadata import fill_template_values
from ppcore.schema.base import BaseSchema
from ppcore.schema.step import StepSchema
from ppcore.utils.helpers import to_list
from ppcore.schema.forecast import (
    Dataset,
    Forecast,
    Reforecast,
    Climatology,
    ForecastDefinition,
    ReforecastDefinition,
    ClimatologyDefinition,
    definition_to_dataset,
)
from ppcore.schema.deriver import ForecastDeriver, ClimatologyDeriver
from ppcore.schema.filters import (
    _members,
    _selection,
    _steplength,
    _steptype,
)
from ppcore.schema.exceptions import PProcInputSchemaError, PProcDatasetError
from ppcore.utils.requests import (
    validate_request,
    update_request,
    expand,
)
from ppcore.utils.mars import extract_mars
from ppcore.utils.qube import union

logger = logging.getLogger(__name__)


INHERIT_FROM_OUTPUT = [
    "class",
    "expver",
    "date",
    "time",
    "levtype",
    "levelist",
    "param",
    "step",
    "model",
    "method",
    "origin",
    "system",
]


class ForecastInput(PProcBaseModel):
    select: dict = {}
    derive: dict[str, list[ForecastDeriver]] = {}
    override: dict = {}
    select_to_override: bool = False

    @field_validator("derive", mode="before")
    @classmethod
    def format_derivers(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key, derivers in data.items():
                if isinstance(derivers, dict):
                    data[key] = [derivers]
        return data

    def _from_output(self, request: dict, pop: list[str] = []) -> dict:
        out = {}
        inherit = set(INHERIT_FROM_OUTPUT)
        for key in self.derive.keys():
            if key in inherit:
                inherit.remove(key)
        for key in inherit:
            if key in request:
                out[key] = request[key]
        return out

    def derived_selection(self, base_request: dict, forecast: Dataset) -> dict:
        derived_selection = {}
        request = base_request.copy()
        for key, derivers in self.derive.items():
            for deriver in derivers:
                request[key] = deriver.derive(request, forecast)
            derived_selection[key] = request[key]
        return derived_selection

    def requests(
        self,
        base_request: dict,
        forecast: Dataset,
        **extra,
    ) -> Iterator[dict]:
        select_criteria = self.create_selection(base_request, forecast)
        selection, select_overrides = forecast.select(
            select_criteria, select_to_override=self.select_to_override
        )
        logger.debug(
            "Selected cubes: %s, select overrides: %s", selection, select_overrides
        )
        override = self.override.copy()
        override.update(select_overrides)
        if "target_grid" in base_request:
            override["target_grid"] = base_request["target_grid"]
        pop = []
        for key, value in override.items():
            if value is None:
                override.pop(key)
                pop.append(key)

        # Order outputs by unperturbed first, if present
        datacubes = []
        if (
            getattr(forecast, "unperturbed", None) is not None
            and selection.select(forecast.unperturbed).n_leaves > 0
        ):
            datacubes.extend(selection.select(forecast.unperturbed).datacubes())
            datacubes.extend(
                cube for cube in selection.datacubes() if cube not in datacubes
            )
        else:
            datacubes = list(selection.datacubes())
        for cube in datacubes:
            cube.update(override)
            for key in pop:
                cube.pop(key, None)
            if cube.get("type", "") in ["fcmean", "fcmax", "fcstdev", "fcmin"]:
                cube.setdefault("number", 0)
            cube.update(extra)
            # Sort request keys for consistent ordering in output configs
            yield validate_request({k: v for k, v in sorted(cube.items())})

    def create_selection(
        self, base_request: dict, forecast: Dataset, include_derived: bool = True
    ) -> dict:
        selection = self.select.copy()
        if include_derived:
            selection.update(self.derived_selection(base_request, forecast))
        selection = fill_template_values(selection, base_request)
        if forecast.wave is not None and "stream" not in selection:
            selection["stream"] = forecast.stream(base_request)
        return validate_request({**self._from_output(base_request), **selection})

    def match(
        self, request: dict, forecast: Union[Forecast, Reforecast]
    ) -> Iterator[Qube]:
        if "param" not in self.select:
            groupby = ["param", "levtype"]
        else:
            groupby = ["levtype"]

        selection = self.create_selection(request, forecast, include_derived=False)
        request_qube, _ = forecast.select(selection)
        logger.debug(
            "Selected %s from forecast using selection %s", request_qube, selection
        )
        seen = set()
        for cube in request_qube.datacubes():
            for selection in zip(*[cube[axis] for axis in groupby if axis in cube]):
                selection_key = str(selection)
                if selection_key in seen:
                    continue
                seen.add(selection_key)
                selected = request_qube.select(dict(zip(groupby, selection)))
                if selected.n_leaves == 0:
                    raise KeyError(
                        f"No datacubes available for selection {selection} from request {request_qube}"
                    )
                yield selected


class ClimatologyInput(ForecastInput):
    derive: dict[str, list[ClimatologyDeriver]] = {}

    @field_validator("select", mode="after")
    @classmethod
    def populate_request(cls, select: dict) -> dict:
        if select.get("type", None) == "cd":
            select["quantile"] = [f"{x}:100" for x in range(0, 101)]
        return select

    @field_validator("derive", mode="after")
    @classmethod
    def validate_derivers(
        cls, derive: dict[str, list[ClimatologyDeriver]]
    ) -> dict[str, list[ClimatologyDeriver]]:
        if "step" not in derive:
            raise PProcInputSchemaError("Step driver required for climatology input")
        if "date" not in derive:
            raise PProcInputSchemaError("Date driver required for climatology input")
        return derive

    def create_selection(
        self, base_request: dict, forecast: Dataset, include_derived: bool = True
    ) -> dict:
        selection = self.select.copy()
        if include_derived:
            selection.update(self.derived_selection(base_request, forecast))
        selection = fill_template_values(selection, base_request)
        if forecast.wave is not None and "stream" not in selection:
            selection["stream"] = forecast.stream(base_request)
        return validate_request(
            {**self._from_output(base_request, pop=["number", "date"]), **selection}
        )


class InputConfig(PProcBaseModel):
    fc_inputs: list[ForecastInput]
    clim_inputs: Optional[list[ClimatologyInput]] = None
    from_inputs: bool = True

    @field_validator("fc_inputs", mode="before")
    @classmethod
    def format_fc_inputs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = [data]
        return data

    @field_validator("clim_inputs", mode="before")
    @classmethod
    def format_clim_inputs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = [data]
        return data

    def inputs(
        self,
        output_request: dict,
        forecast: Union[Forecast],
        climatology: Optional[Climatology] = None,
    ) -> Iterator[dict]:
        sample_input = None
        for inp in self.fc_inputs:
            inputs = list(inp.requests(output_request, forecast))
            if len(inputs) == 0:
                raise PProcDatasetError(
                    f"No forecast inputs matched for {output_request}, requiring {inp}"
                )
            if sample_input is None:
                sample_input = inputs[0]
            yield from inputs

        if self.clim_inputs:
            assert (
                climatology is not None
            ), "Climatology input defined but no climatology provided"
            for inp in self.clim_inputs:
                yield from inp.requests(
                    validate_request(sample_input), climatology, climatology=True
                )

    def matched_outputs(
        self,
        request: dict,
        forecast: Union[Forecast, Reforecast],
        climatology: Optional[Climatology] = None,
        step_schema: Optional[StepSchema] = None,
    ) -> Iterator[dict]:
        fc_inputs = [
            union(paired_inputs)
            for paired_inputs in zip(
                *[list(inp.match(request, forecast)) for inp in self.fc_inputs]
            )
        ]
        if len(fc_inputs) == 0:
            logger.debug("No inputs for request %s", request)
            return iter([])

        if self.clim_inputs and len(fc_inputs) > 1:
            raise PProcDatasetError("Climatology should be uniquely tied to forecast")
        # Determine output steps that can be generated
        step_schema = step_schema or StepSchema({})
        fc_steps = list(fc_inputs[0].axes()["step"])
        if len(fc_steps) > 1:
            if any("-" in str(x) for x in fc_steps):
                raise NotImplementedError(
                    "Combining different step ranges is not supported"
                )
            fc_steps = sorted(list(map(int, fc_steps)))

        sample = forecast.sample_datacube(fc_inputs[0], pop=["step", "number"])
        step_dim, out_steps = step_schema.out_steps({**sample, **request}, fc_steps)
        logger.debug(
            "Obtained values %s from forecast steps %s for dim %s",
            out_steps,
            fc_steps,
            step_dim,
        )

        for fc_input, step in itertools.product(fc_inputs, out_steps):
            sample_datacube = forecast.sample_datacube(fc_input, pop=["step", "number"])
            out = {**sample_datacube, **request, step_dim: step}
            # Determine climatology is present
            if self.clim_inputs:
                if climatology is None:
                    raise PProcDatasetError(
                        "Climatology input required but no climatology provided"
                    )
                clim_input = [
                    union(paired_inputs)
                    for paired_inputs in zip(
                        *[
                            list(inp.match(request, climatology))
                            for inp in self.clim_inputs
                        ]
                    )
                ]
                if len(clim_input) == 0:
                    continue
                if len(clim_input) > 1:
                    raise PProcDatasetError(
                        "Climatology should be uniquely tied to forecast"
                    )
            yield out


def _update_inputs(config, update) -> dict:
    for key, val in update.items():
        config[key] = update_request(config.get(key, {}), val)
    return config


class InputSchema(BaseSchema):
    exception = PProcInputSchemaError
    custom_update = {
        "fc_inputs": _update_inputs,
        "clim_inputs": _update_inputs,
    }
    custom_filter = {
        "steptype": _steptype,
        "steplength": _steplength,
        "selection": _selection,
        "members": _members,
    }

    def inputs(
        self,
        output_request: dict,
        forecast: Union[ForecastDefinition, ReforecastDefinition],
        climatology: Optional[Climatology] = None,
    ) -> Iterator[dict]:
        output_request = validate_request(output_request)
        config = InputConfig(**self.traverse(output_request))
        forecast = definition_to_dataset(forecast)
        if climatology is not None:
            climatology = definition_to_dataset(climatology)

        for inp in config.inputs(output_request, forecast, climatology):
            yield inp

    def _set_defaults(cls, output_request: dict, input_requests: list[dict]) -> dict:
        output_request = validate_request(output_request)
        tp = output_request["type"]
        if tp in ["fcmean", "fcmax", "fcstdev", "fcmin"]:
            output_request["number"] = sum(
                [to_list(req.get("number", [0])) for req in input_requests], []
            )
        elif tp in ["pf", "gwt"]:
            for req in input_requests:
                if req["type"] in ["pf", "gwt"] and "number" in req:
                    output_request["number"] = req["number"]
                    break
        elif tp in ["pb", "cd"]:
            output_request.setdefault("quantile", [f"{x}:100" for x in range(0, 101)])
        elif tp == "sot":
            output_request.setdefault("number", [10, 90])
        return output_request

    def outputs(
        self,
        forecast: Union[ForecastDefinition, ReforecastDefinition],
        climatology: Optional[ClimatologyDefinition] = None,
        step_schema: Optional[StepSchema] = None,
        output_template: Optional[dict[str, Any]] = None,
        method: Literal["dfs", "bfs"] = "bfs",
        enable_cache: bool = True,
    ) -> Iterator[tuple[dict, list[dict]]]:
        """
        Assumes inputs are from the same forecast for a single date and time
        """
        output_template = output_template or {}
        forecast = definition_to_dataset(forecast)
        if climatology is not None:
            climatology = definition_to_dataset(climatology)
        # Check that the output template contains the required keys to derive outputs from inputs
        # To avoid generating outputs across different streams
        # TODO: consider splitting schema by dataset definers to avoid this requirement
        dataset_definers = {"class", "stream", "model"}
        required = set.intersection(dataset_definers, self.all_filters)
        if any(key not in output_template for key in required):
            raise ValueError(
                f"Output template must contain {required} to derive outputs from inputs"
            )
        for template in expand(output_template, dim=list(self.all_filters) + ["step"]):
            template = validate_request(template)
            # Derive output requests and corresponding input configs that match the template
            for base_output, config in self.reconstruct(
                output_template=template,
                from_inputs=True,
                method=method,
                enable_cache=enable_cache,
            ):
                logger.debug("Reconstructed output %s", base_output)
                config = InputConfig(**config)
                # Perform intersection with the forecast and climatology qubes to determine
                # the actual output requests that can be generated from the inputs
                for mout in config.matched_outputs(
                    base_output, forecast, climatology, step_schema
                ):
                    inputs = list(config.inputs(mout, forecast, climatology))
                    mout = extract_mars(self._set_defaults(mout, inputs))
                    logger.info("Output %s, requiring inputs %s", mout, inputs)
                    yield mout, inputs
