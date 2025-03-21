from typing import Optional, Union, Iterator
from typing_extensions import Self
from pydantic import BaseModel, model_validator
import copy
import pandas as pd
import logging
import os

from pproc.schema.base import Schema
from pproc.schema.deriver import (
    ForecastStepDeriver,
    DefaultStepDeriver,
    ClimDateDeriver,
    ClimStepDeriver,
)
from pproc.schema.step import StepSchema
from pproc.config.utils import update_request, expand, squeeze, deep_update

logging.getLogger("pproc").setLevel(os.environ.get("PPROC_LOG", "INFO").upper())
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def format_request(request: dict) -> dict:
    for key, value in request.items():
        if isinstance(value, list) and len(value) == 1:
            request[key] = value[0]
        if key == "number":
            request[key] = list(map(int, value))
    return request


class ForecastInput(BaseModel):
    members: Optional[dict] = None
    request: dict
    derive_step: ForecastStepDeriver

    @model_validator(mode="after")
    def populate_request(self) -> Self:
        if self.members:
            self.request.setdefault(
                "number", list(range(self.members["start"], self.members["end"] + 1))
            )
        else:
            self.request.pop("number", None)
        return self

    def populate_derived(self, output_request, fc_steps: list[int]):
        for k, v in self.request.items():
            self.request[k] = v if v != f"{{{k}}}" else output_request[k]
        self.request["step"] = self.derive_step.derive(output_request, fc_steps)


class ClimatologyInput(ForecastInput):
    members: Optional[dict] = None
    request: dict
    derive_date: Optional[ClimDateDeriver] = None
    derive_step: Optional[ClimStepDeriver] = None

    @model_validator(mode="after")
    def populate_request(self) -> Self:
        super().populate_request()
        if self.request.get("type", None) == "cd":
            self.request["quantile"] = [f"{x}:100" for x in range(0, 101)]
        return self

    def populate_derived(
        self, fc_request: dict, clim_steps: list[int | str], scheme: str
    ):
        assert (
            self.derive_step and self.derive_date
        ), "Both step and date derivers required"
        for k, v in self.request.items():
            self.request[k] = v if v != f"{{{k}}}" else fc_request[k]
        self.request["step"] = self.derive_step.derive(fc_request, clim_steps)
        self.request["date"] = self.derive_date.derive(fc_request, scheme)


class ForecastConfig(BaseModel):
    inputs: list[ForecastInput]

    def steps(self) -> list[int]:
        out = set()
        for inp in self.inputs:
            steps = inp.request["step"]
            assert isinstance(steps, list), f"Expected list of steps, got {steps}"
            out.update(steps)
        out = list(out)
        out.sort()
        return out

    def base_request(self, *exclude) -> dict:
        out = {}
        if len(self.inputs) == 0:
            return out

        for key in set.intersection(*[set(inp.request.keys()) for inp in self.inputs]):
            if all(
                [inp.request[key] == self.inputs[0].request[key] for inp in self.inputs]
            ):
                if key not in exclude:
                    out[key] = self.inputs[0].request[key]
        return out

    def match(self, input_requests: list[dict]) -> Iterator[Self]:
        paramless = ["param" not in inp.request for inp in self.inputs]
        if any(paramless):
            assert all(paramless), "All or none of the inputs must specify param"
            require = sum([list(expand(inp.request)) for inp in self.inputs], [])
            intersection = _intersect(require, input_requests)
            for _, pgroup in intersection.groupby("param", sort=False):
                # Squeeze certain back together into a single request
                squeeze_dims = ["step", "number", "levelist"]
                reqs = list(squeeze(pgroup.to_dict("records"), squeeze_dims))
                reqs.sort(key=lambda x: x["type"])
                yield type(self)(
                    **self.model_dump(exclude=("inputs",), by_alias=True),
                    inputs=[
                        {
                            **inp.model_dump(exclude=("request",), by_alias=True),
                            "request": reqs[i],
                        }
                        for i, inp in enumerate(self.inputs)
                    ],
                )
        else:
            updated_reqs = []
            for inp in self.inputs:
                require = expand(inp.request)
                intersection = _intersect(require, input_requests)
                squeeze_dims = ["step", "number", "levelist", "param", "quantile"]
                reqs = list(squeeze(intersection.to_dict("records"), squeeze_dims))
                assert len(reqs) == 1
                updated_reqs.append(reqs[0])
            yield type(self)(
                **self.model_dump(exclude=("inputs",), by_alias=True),
                inputs=[
                    {
                        **inp.model_dump(exclude=("request",), by_alias=True),
                        "request": updated_reqs[i],
                    }
                    for i, inp in enumerate(self.inputs)
                ],
            )


class ClimatologyConfig(ForecastConfig):
    scheme: str = ""
    inputs: list[ClimatologyInput]
    required: bool = False

    def match(self, input_requests: list[dict]) -> Iterator[Self]:
        assert all(["param" in inp.request for inp in self.inputs])
        yield from super().match(input_requests)


class InputConfig(BaseModel):
    forecast: ForecastConfig
    climatology: ClimatologyConfig
    from_inputs: bool = True

    def populate_derived(
        self,
        output_request: dict,
        fc_steps: list[int],
        clim_steps: Optional[list[int | str]] = None,
    ):
        for input in self.forecast.inputs:
            input.populate_derived(output_request, fc_steps)

        if self.climatology.required:
            for input in self.climatology.inputs:
                input.populate_derived(
                    fc_request=self.forecast.base_request(),
                    clim_steps=clim_steps,
                    scheme=self.climatology.scheme,
                )

    def inputs(self) -> Iterator[dict]:
        for input in self.forecast.inputs:
            yield format_request(input.request)
        if self.climatology.required:
            for input in self.climatology.inputs:
                yield format_request(input.request)

    def match(self, input_requests: list[dict]) -> Iterator[Self]:
        fc_inputs = list(self.forecast.match(input_requests))

        if self.climatology.required:
            assert (
                len(fc_inputs) == 1
            ), "Climatology should be uniquely tied to forecast"

            fc_config = fc_inputs[0]
            for clim_inp in self.climatology.inputs:
                clim_inp.request["date"] = clim_inp.derive_date.derive(
                    fc_config.base_request(), self.climatology.scheme
                )
            try:
                clim_inputs = list(self.climatology.match(input_requests))
            except AssertionError:
                logger.warning("Could not find required climatology in requests")
                return iter([])

            yield InputConfig(
                forecast=fc_inputs[0],
                climatology=clim_inputs[0],
            )
        else:
            for fc_input in fc_inputs:
                yield InputConfig(
                    forecast=fc_input,
                    climatology=self.climatology,
                    from_inputs=self.from_inputs,
                )


def _steptype(request: dict, key: str) -> str:
    step = request.get("step", [])
    steprange = str(step).split("-")
    return "range" if len(steprange) == 2 else "instantaneous"


def _update_config(config: dict, update: dict[str, dict]) -> dict:
    fc_type = list(update.keys())[0]
    fc_update = list(update.values())[0].copy()
    fc_config = config[fc_type].model_dump(exclude_none=True, by_alias=True)
    current_inputs = fc_config.pop("inputs")
    update_inputs = fc_update.pop("inputs", [])
    inputs = update_request(current_inputs, update_inputs)
    config[fc_type] = type(config[fc_type])(
        **deep_update(fc_config, fc_update), inputs=inputs
    )
    return config


def _intersect(
    lista: list[dict], listb: list[dict], how: str = "inner"
) -> pd.DataFrame:
    dfa = pd.DataFrame(lista)
    dfa.drop(columns="number", errors="ignore", inplace=True)
    dfa.drop_duplicates(inplace=True)
    dfb = pd.DataFrame(listb)
    merged = dfa.merge(dfb, how=how, on=dfa.columns.tolist())

    num_expected = len(dfa)
    restricted = merged.drop(
        columns=["number"] + [x for x in dfb.columns if x not in dfa.columns],
        errors="ignore",
    )
    restricted.drop_duplicates(inplace=True)
    return merged if len(restricted) == num_expected else pd.DataFrame()


def _match_forecast(
    out: dict, forecast: Union[ForecastConfig, ClimatologyConfig], match: dict
) -> bool:
    for key, value in match.items():
        if key == "inputs":
            if not getattr(forecast, "required", True) or len(forecast.inputs) == 0:
                continue
            require = []
            for inp in forecast.inputs:
                for key in list(inp.request.keys()):
                    if inp.request[key] == f"{{{key}}}":
                        if key in out:
                            inp.request[key] = out[key]
                        else:
                            return True
                require.extend(expand(inp.request))
            merged = _intersect(require, value)
            if len(merged) == 0:
                return False
        elif getattr(forecast, key, value) != value:
            return False
    return True


class InputSchema(Schema):
    custom_update = {
        "forecast": _update_config,
        "climatology": _update_config,
    }
    custom_filter = {"steptype": _steptype}
    custom_match = {"forecast": _match_forecast, "climatology": _match_forecast}

    @classmethod
    def _format_output_request(
        cls, request: dict, pop: Optional[list[str]] = []
    ) -> dict:
        out = copy.deepcopy(request)
        # Remove number from output types e.g. sot where number is not
        # associated with ensemble number
        if request["type"] not in ["pf", "fcmean", "fcmax", "fcstdev", "fcmin", "fc"]:
            out.pop("number", None)
        if request["type"] == "sot":
            out["number"] = [10, 90]
        if request["type"] == ["pb", "cd"]:
            out["quantile"] = [f"{x}:100" for x in range(0, 101)]
        for key in pop + ["step", "fcmonth"]:
            out.pop(key, None)
        return format_request(out)

    def inputs(self, output_request: dict, step_schema: StepSchema) -> Iterator[dict]:
        initial = {
            "forecast": ForecastConfig(
                inputs=[
                    {
                        "request": self._format_output_request(output_request),
                        "derive_step": DefaultStepDeriver(),
                    }
                ],
            ),
            "climatology": ClimatologyConfig(
                inputs=[
                    {
                        "request": self._format_output_request(
                            output_request, pop=["date"]
                        )
                    }
                ],
            ),
        }
        config = InputConfig(**self.traverse(output_request, initial))
        fc_steps = step_schema.in_steps(output_request)
        clim_steps = (
            None
            if not config.climatology.required
            else step_schema.out_steps(config.climatology.inputs[0].request)[1]
        )
        config.populate_derived(output_request, fc_steps, clim_steps)
        yield from config.inputs()

    def reconstruct(
        self, output_template: Optional[dict] = None, **match
    ) -> Iterator[tuple[dict, InputConfig]]:
        for cfg in self._find_matching(
            self.schema,
            [
                {
                    "recon_req": output_template or {},
                    "forecast": ForecastConfig(inputs=[]),
                    "climatology": ClimatologyConfig(inputs=[]),
                }
            ],
            **match,
        ):
            out, input_config = cfg.pop("recon_req"), InputConfig(**cfg)
            for req in input_config.inputs():
                if req["param"] == "{param}":
                    req.pop("param")
            logger.info("Reconstructed output %s, with config %s", out, input_config)
            yield out, input_config

    def _set_defaults(cls, output_request: dict, input_requests: list[dict]) -> dict:
        tp = output_request["type"]
        if tp in ["fcmean", "fcmax", "fcstdev", "fcmin"]:
            output_request["number"] = sum(
                [req.get("number", [0]) for req in input_requests], []
            )
        elif tp == "pf":
            for req in input_requests:
                if req["type"] == "pf":
                    output_request["number"] = req["number"]
                    break
        elif tp in ["pb", "cd"]:
            output_request.setdefault("quantile", [f"{x}:100" for x in range(0, 101)])
        elif tp == "sot":
            output_request.setdefault("number", [10, 90])
        return format_request(output_request)

    def outputs(
        self,
        input_requests: list[dict],
        step_schema: Optional[StepSchema] = None,
        output_template: Optional[dict] = None,
    ) -> Iterator[tuple[dict, list[dict]]]:
        """
        Assumes inputs are from the same forecast for a single date and time
        """
        for base_output, config in self.reconstruct(
            output_template=format_request(output_template),
            forecast={"inputs": input_requests},
            climatology={"inputs": input_requests},
            from_inputs=True,
        ):
            for mconfig in config.match(input_requests):
                logger.debug("Matched config: %s", mconfig)
                mout = {
                    **mconfig.forecast.base_request("step", "number"),
                    **base_output,
                }

                step_schema = step_schema or StepSchema({})
                step_dim, out_steps = step_schema.out_steps(
                    mout, mconfig.forecast.steps()
                )
                clim_steps = (
                    None
                    if not mconfig.climatology.required
                    else mconfig.climatology.steps()
                )

                for step in out_steps:
                    out = {**mout, step_dim: step}
                    sconfig = mconfig.model_copy(deep=True)
                    sconfig.populate_derived(out, sconfig.forecast.steps(), clim_steps)
                    out, inputs = out, list(sconfig.inputs())
                    logger.info("Output %s, requiring inputs %s", out, inputs)
                    yield self._set_defaults(out, inputs), inputs
