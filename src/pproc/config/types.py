import copy
from typing import Optional, List, Any, Annotated, Iterator
from typing_extensions import Self, Union
from pydantic import model_validator, Field, Tag, Discriminator
import numpy as np
import datetime

from conflator import CLIArg


from pproc.config.base import BaseConfig, Parallelisation
from pproc.config import io
from pproc.config.param import ParamConfig, partial_equality
from pproc.config.utils import _set, _get, update_request, deep_update
from pproc.common.stepseq import steprange_to_fcmonth
from pproc.extremes.indices import Index, SUPPORTED_INDICES, create_indices


def steprange(steps: list[int] | str) -> str:
    if isinstance(steps, str):
        return steps
    return f"{steps[0]}-{steps[-1]}"


def end_step(step: int | str) -> int:
    return step if isinstance(step, int) else int(step.split("-")[1])


class EnsmsConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.EnsmsOutputModel = io.EnsmsOutputModel()


class QuantilesConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.QuantilesOutputModel = io.QuantilesOutputModel()
    quantiles: int | List[float] = 100
    _total_number: int = 0
    _even_spacing: bool = None

    @property
    def even_spacing(self) -> bool:
        if self._even_spacing is None:
            self._even_spacing = isinstance(self.quantiles, int) or np.all(
                np.diff(self.quantiles) == self.quantiles[1] - self.quantiles[0]
            )
        return self._even_spacing

    @property
    def total_number(self) -> int:
        if self._total_number == 0:
            num_quantiles = (
                self.quantiles
                if isinstance(self.quantiles, int)
                else (len(self.quantiles) - 1)
            )
            self._total_number = num_quantiles if self._even_spacing else 100
        return self._total_number

    def quantile_indices(self, index: int) -> List[int]:
        pert_number = index if self.even_spacing else int(self.quantiles[index] * 100)
        return pert_number, self.total_number

    @classmethod
    def from_schema(cls, schema_config: dict, **overrides) -> Self:
        quantiles = schema_config.get("quantiles", None)
        return super().from_schema(schema_config, **overrides, quantiles=quantiles)

    def _format_out(self, param: ParamConfig, req) -> dict:
        req = super()._format_out(param, req)
        num_quantiles = (
            self.quantiles
            if isinstance(self.quantiles, int)
            else (len(self.quantiles) - 1)
        )
        req["quantile"] = [
            f"{qindices[0]}:{qindices[1]}"
            for qindices in map(self.quantile_indices, range(num_quantiles + 1))
        ]
        return req


class AccumParamConfig(ParamConfig):
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    out_accum_key: str = "perturbationNumber"
    out_accum_values: Optional[list[float]] = None


class AccumConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.AccumOutputModel = io.AccumOutputModel()
    parameters: list[AccumParamConfig]

    def _format_out(self, param: AccumParamConfig, req: dict) -> dict:
        req = req.copy()
        if req["type"] not in ["fcmean", "fcmax", "fcstdev", "fcmin"]:
            return req

        src_name = self.sources.names[0]
        source = param.in_sources(self.sources, src_name)
        src_reqs = source[0].request
        if isinstance(src_reqs, dict):
            src_reqs = [src_reqs]

        number = None
        num_members = super().compute_totalfields(src_name)
        for src_req in src_reqs:
            if len(src_req) == 0:
                continue
            number = src_req.get("number", number)

        if len(number) == num_members - 1:
            number = [0] + number
        req["number"] = number
        return req


class MonthlyStatsConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.MonthlyStatsOutputModel = io.MonthlyStatsOutputModel()
    parameters: list[AccumParamConfig]

    def _format_out(self, param: ParamConfig, req: dict) -> dict:
        req = req.copy()
        step_ranges = req.pop("step")
        date = datetime.datetime.strptime(str(req["date"]), "%Y%m%d")
        fcmonths = [
            steprange_to_fcmonth(date, step_range) for step_range in step_ranges
        ]
        req["fcmonth"] = fcmonths
        return req


class HistParamConfig(ParamConfig):
    bins: List[float]
    mod: Optional[int] = None
    normalise: bool = True
    scale_out: Optional[float] = None

    def _format_out(self, param: ParamConfig, req: dict) -> dict:
        req = super()._format_out(param, req)
        req["quantile"] = [f"{x}:{len(self.bins)}" for x in range(1, len(self.bins))]
        return req


class HistogramConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    outputs: io.HistogramOutputModel = io.HistogramOutputModel()
    parameters: list[HistParamConfig]


class ClimParamConfig(ParamConfig):
    clim: ParamConfig
    _merge_exclude = ("accumulations", "sources", "clim")

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: Any) -> Any:
        clim = _get(data, "clim", {})
        if isinstance(clim, dict):
            clim_options = {**data, **clim}
            _set(data, "clim", ParamConfig(**clim_options))
        return data

    def in_keys(
        self, sources: io.SourceCollection, filters: Optional[list[str]] = None
    ) -> Iterator[dict]:
        for source in sources.names:
            for psource in self.in_sources(sources, source):
                if filters and psource.type not in filters:
                    continue

                reqs = (
                    psource.request
                    if isinstance(psource.request, list)
                    else [psource.request]
                )
                for req in reqs:
                    req["source"] = (
                        psource.path if psource.path is not None else psource.type
                    )
                    if isinstance(req.get("step", []), dict):
                        req["step"] = list(req["step"].values())

                    accum_updates = (
                        getattr(self, source).accumulations
                        if hasattr(self, source)
                        else {}
                    )
                    accumulations = deep_update(
                        self.accumulations.copy(), accum_updates
                    )
                    req.update(
                        {
                            key: accum.unique_coords()
                            for key, accum in self.accumulations.items()
                            if key not in req
                        }
                    )
                    yield req

    def _merge_sources(self, other: Self) -> dict:
        new_sources = copy.deepcopy(self.sources)
        other_sources = copy.deepcopy(other.sources)
        if "clim" in new_sources:
            if "clim" not in other_sources:
                raise ValueError("Merging of sources requires same source types")
            steps = []
            for source in [new_sources, other_sources]:
                clim_request = source["clim"].get("request", {})
                if isinstance(clim_request, list):
                    clim_request = clim_request[0]
                if clim_steps := clim_request.get("step", {}):
                    steps.append(clim_steps)
            if len(steps) > 0:
                if {**steps[0], **steps[1]} != {**steps[1], **steps[0]}:
                    raise ValueError(
                        "Merging of two parameter configs requires clim steps to be compatible"
                    )
                [
                    update_request(
                        source["clim"].get("request", {}),
                        {"step": {**steps[0], **steps[1]}},
                    )
                    for source in [new_sources, other_sources]
                ]
        if new_sources != other_sources:
            raise ValueError(
                "Merging of sources requires equality, except for clim steps"
            )
        return new_sources


class SigniParamConfig(ClimParamConfig):
    clim_em: ParamConfig
    epsilon: Optional[float] = None
    epsilon_is_abs: bool = True
    _merge_exclude = ("accumulations", "clim", "clim_em")

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: Any) -> Any:
        clim = _get(data, "clim", {})
        if isinstance(clim, dict):
            clim_options = {**data, **clim}
            _set(data, "clim", ParamConfig(**clim_options))
        clim_em = _get(data, "clim_em", {})
        if isinstance(clim_em, dict):
            if len(clim_em) > 0:
                clim_options = {**data, **clim_em}
            else:
                clim_options = {**data, **clim}
            _set(data, "clim_em", ParamConfig(**clim_options))
        return data


class SigniConfig(BaseConfig):
    clim_total_fields: Annotated[int, Field(validate_default=True)] = 0
    parallelisation: Parallelisation = Parallelisation()
    sources: io.SignificanceSourceModel
    outputs: io.SignificanceOutputModel = io.SignificanceOutputModel()
    parameters: list[SigniParamConfig]
    use_clim_anomaly: Annotated[
        bool,
        CLIArg("--use-clim-anomaly", action="store_true", default=None),
        Field(description="Use anomaly of climatology in significance computation"),
    ] = False

    @model_validator(mode="after")
    def validate_totalfields(self) -> Self:
        super().validate_totalfields()
        if self.clim_total_fields == 0:
            self.clim_total_fields = self.total_fields("clim")
        return self

    classmethod

    def _populate_sources(
        cls, inputs: list[dict], accum_dims: list[str], **overrides
    ) -> dict:
        sorted_requests = {}
        for inp in inputs:
            if inp["type"] == "fcmean":
                src_name = "clim"
            elif inp["type"] == "taem":
                src_name = "clim_em"
            else:
                src_name = "fc"
            [inp.pop(dim, None) for dim in accum_dims]
            sorted_requests.setdefault(src_name, []).append(inp)

        sources = {}
        for src_name, requests in sorted_requests.items():
            src_overrides = overrides.get(src_name, {})
            request_overrides = src_overrides.pop("request", {})
            updated_inputs = update_request(requests, request_overrides)
            sources[src_name] = {
                "request": updated_inputs
                if len(updated_inputs) > 1
                else updated_inputs[0],
                **src_overrides,
            }
        return sources


class AnomalyConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    sources: io.ClimSourceModel
    outputs: io.AnomalyOutputModel = io.AnomalyOutputModel()
    parameters: list[ClimParamConfig]

    @classmethod
    def _populate_sources(
        cls, inputs: list[dict], accum_dims: list[str], **overrides
    ) -> dict:
        sorted_requests = {}
        for inp in inputs:
            if inp["type"] == "taem":
                src_name = "clim"
            else:
                src_name = "fc"
            [inp.pop(dim, None) for dim in accum_dims]
            sorted_requests.setdefault(src_name, []).append(inp)

        sources = {}
        for src_name, requests in sorted_requests.items():
            src_overrides = overrides.get(src_name, {})
            request_overrides = src_overrides.pop("request", {})
            updated_inputs = update_request(requests, request_overrides)
            sources[src_name] = {
                "request": updated_inputs
                if len(updated_inputs) > 1
                else updated_inputs[0],
                **src_overrides,
            }
        return sources


def anom_discriminator(config: Any) -> str:
    clim = _get(config, "clim", None)
    return "clim" if clim else "base"


class ProbParamConfig(ClimParamConfig):
    clim: Optional[ParamConfig] = None

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: Any) -> Any:
        clim = _get(data, "clim", None)
        if isinstance(clim, dict):
            clim_options = {
                **data,
                "preprocessing": [],
                "accumulations": {},
                "metadata": {},
                **clim,
            }
            _set(data, "clim", ParamConfig(**clim_options))
        return data

    def _merge_clim(self, other: Self) -> None:
        return None


class ProbConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    sources: Annotated[
        Union[
            Annotated[io.BaseSourceModel, Tag("base")],
            Annotated[io.ClimSourceModel, Tag("clim")],
        ],
        Discriminator(anom_discriminator),
    ]
    outputs: io.ProbOutputModel = io.ProbOutputModel()
    parameters: list[ProbParamConfig]

    @model_validator(mode="after")
    def validate_param(self) -> Self:
        if isinstance(self.sources, io.ClimSourceModel):
            for param in self.parameters:
                if not param.clim:
                    param.clim = ParamConfig(
                        **param.model_dump(
                            exclude=("preprocessing", "accumulations", "metadata"),
                            by_alias=True,
                        ),
                        accumulations={},
                    )
        return self

    @classmethod
    def from_schema(cls, schema_config: dict, **overrides) -> Self:
        schema_config = copy.deepcopy(schema_config)
        threshold = schema_config.pop("threshold")
        threshold["out_paramid"] = schema_config["metadata"].pop("paramId")
        schema_config["accumulations"]["step"]["thresholds"] = [threshold]
        return super().from_schema(schema_config, **overrides)

    @classmethod
    def _populate_sources(
        cls, inputs: list[dict], accum_dims: list[str], **overrides
    ) -> dict:
        sorted_requests = {}
        fc_step = 0
        clim_step = 0
        for inp in inputs:
            if inp["type"] in ["em", "es"]:
                src_name = "clim"
                clim_step = inp["step"]
            else:
                src_name = "fc"
                fc_step = inp["step"]
            [inp.pop(dim, None) for dim in accum_dims]
            sorted_requests.setdefault(src_name, []).append(inp)

        for clim_inp in sorted_requests.get("clim", []):
            clim_inp["step"] = {fc_step: clim_step}

        sources = {}
        for src_name, requests in sorted_requests.items():
            src_overrides = overrides.get(src_name, {})
            request_overrides = src_overrides.pop("request", {})
            updated_inputs = update_request(requests, request_overrides)
            sources[src_name] = {
                "request": updated_inputs
                if len(updated_inputs) > 1
                else updated_inputs[0],
                **src_overrides,
            }
        return sources


class ExtremeParamConfig(ClimParamConfig):
    eps: float = -1.0
    sot: list[int] = []
    cpf_eps: Optional[float] = None
    cpf_symmetric: bool = False
    compute_indices: list[str] = ["efi", "sot"]
    allow_grib1_to_grib2: bool = False
    _merge_exclude: tuple[str] = (
        "accumulations",
        "sources",
        "clim",
        "sot",
        "cpf_eps",
        "compute_indices",
    )

    @model_validator(mode="after")
    def validate_indices(self) -> Self:
        for index in self.compute_indices:
            if index not in SUPPORTED_INDICES:
                raise ValueError(
                    f"Unsupported index {index}. Supported indices are {SUPPORTED_INDICES}"
                )
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: Any) -> Any:
        clim = _get(data, "clim", {})
        if isinstance(clim, dict):
            clim_options = {
                **data,
                "preprocessing": [],
                "accumulations": {},
                "metadata": {},
                **clim,
            }
            _set(data, "clim", ParamConfig(**clim_options))
        return data

    @property
    def indices(self) -> dict[str, Index]:
        return create_indices(self.compute_indices, self.model_dump())

    def out_keys(self, sources: io.SourceCollection) -> Iterator:
        base_outs = [req for req in super().out_keys(sources)]
        indices = self.compute_indices.copy()
        if np.any([x["type"] in ["cf", "fc"] for x in base_outs]):
            indices.append("efic")
        req = base_outs[0].copy()
        for index in indices:
            if index == "sot" and len(self.sot) == 0:
                continue
            req["type"] = index
            yield req

    def _merge_clim(self, other: Self) -> dict:
        return {}

    def _merge_cpf_eps(self, other: Self) -> Optional[float]:
        if self.cpf_eps is None:
            return other.cpf_eps

        if other.cpf_eps is not None and other.cpf_eps != self.cpf_eps:
            raise ValueError(
                "Merging of parameter configs requires cpf_eps to be equal"
            )

        return self.cpf_eps


class ExtremeConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    sources: io.ClimSourceModel
    outputs: io.ExtremeOutputModel = io.ExtremeOutputModel()
    parameters: list[ExtremeParamConfig]

    def _format_out(self, param: ParamConfig, req: dict) -> dict:
        req = super()._format_out(param, req)
        if req["type"] == "sot":
            req["number"] = param.sot
        return req

    @classmethod
    def _populate_sources(
        cls, inputs: list[dict], accum_dims: list[str], **overrides
    ) -> dict:
        sorted_requests = {}
        fc_step = 0
        clim_step = 0
        for inp in inputs:
            if inp["type"] == "cd":
                src_name = "clim"
                clim_step = inp["step"]
            else:
                src_name = "fc"
                fc_step = steprange(inp["step"])
            [inp.pop(dim, None) for dim in accum_dims]
            sorted_requests.setdefault(src_name, []).append(inp)

        for clim_inp in sorted_requests.get("clim", []):
            clim_inp["step"] = {fc_step: clim_step}

        sources = {}
        for src_name, requests in sorted_requests.items():
            src_overrides = overrides.get(src_name, {})
            request_overrides = src_overrides.pop("request", {})
            updated_inputs = update_request(requests, request_overrides)
            sources[src_name] = {
                "request": updated_inputs
                if len(updated_inputs) > 1
                else updated_inputs[0],
                **src_overrides,
            }
        return sources


class WindConfig(BaseConfig):
    parallelisation: int = 1
    outputs: io.WindOutputModel = io.WindOutputModel()
    parameters: list[ParamConfig]

    def _format_out(self, param: ParamConfig, req: dict) -> dict:
        req = req.copy()
        if req["type"] in ["em", "es"]:
            req.pop("number", None)
        return req


class ThermoParamConfig(ParamConfig):
    out_params: list[str | int]

    def in_keys(
        self, sources: io.SourceCollection, filters: Optional[list[str]] = None
    ) -> Iterator[dict]:
        for source in sources.names:
            for psource in self.in_sources(sources, source):
                if filters and psource.type not in filters:
                    continue

                reqs = (
                    psource.request
                    if isinstance(psource.request, list)
                    else [psource.request]
                )
                for req in reqs:
                    req["source"] = (
                        psource.path if psource.path is not None else psource.type
                    )
                    req.update(
                        {
                            key: accum.unique_coords()
                            for key, accum in self.accumulations.items()
                        }
                    )
                    # Override step for instantaneous params, which is equal to output steps
                    if source == "inst":
                        req["step"] = list(self.out_keys(sources))[0]["step"]
                    yield req

    def out_keys(self, sources: io.SourceCollection) -> Iterator:
        for req in super().out_keys(sources):
            req["param"] = self.out_params
            req["step"] = [end_step(x) for x in req["step"]]
            yield req

    def _merge_sources(self, other: Self) -> dict:
        # inst + inst -> merge accums and input params
        # accum + accum -> merge accums and input params
        # inst + (inst, accum) -> only merge in inst steps are encompassed in accum step ranges, becomes accum
        new_sources = self.sources.copy()
        other_sources = other.sources.copy()
        for key in new_sources:
            if key in other_sources:
                current_params = new_sources[key]["request"].pop("param")
                other_params = other_sources[key]["request"].pop("param")
                if not isinstance(current_params, list):
                    params = [current_params]
                if not isinstance(other_params, list):
                    other_params = [other_params]
                if new_sources[key] != other_sources[key]:
                    raise ValueError(
                        "Only sources equal up to request param can be merged"
                    )
                new_sources[key]["request"]["param"] = current_params + [
                    x for x in other_params if x not in current_params
                ]
        for key in other_sources:
            if key not in new_sources:
                new_sources[key] = other_sources[key]
        return new_sources

    def can_merge(self, other: Self) -> bool:
        if self.out_params == other.out_params:
            return True
        if self.accumulations == other.accumulations:
            return True
        out_steps = sum(
            [
                [end_step(x) for x in steps["step"]]
                for steps in self.accumulations["step"].out_mars("step")
            ],
            [],
        )
        other_steps = sum(
            [
                [end_step(x) for x in steps["step"]]
                for steps in other.accumulations["step"].out_mars("step")
            ],
            [],
        )
        return out_steps == other_steps

    def merge(self, other: Self) -> Self:
        if self.out_params == other.out_params:
            return super().merge(other)
        exclude = ("name", "accumulations", "out_params", "sources")
        if not partial_equality(self, other, exclude=exclude):
            raise ValueError(
                f"Merging of two parameter configs requires equality, except for {exclude}"
            )

        merged = self.model_dump(by_alias=True, exclude=exclude)
        if (
            self.accumulations != other.accumulations
            and self.accumulations["step"].operation != "difference"
        ):
            merged["accumulations"] = other.accumulations
        else:
            merged["accumulations"] = self.accumulations
        merged["out_params"] = self.out_params + [
            x for x in other.out_params if x not in self.out_params
        ]
        merged["sources"] = self._merge_sources(other)
        merged["name"] = self.name
        return type(self)(**merged)


class ThermoConfig(BaseConfig):
    parallelisation: Parallelisation = Parallelisation()
    sources: io.ThermoSourceModel
    outputs: io.ThermoOutputModel = io.ThermoOutputModel()
    parameters: list[ThermoParamConfig]
    validateutci: bool = False
    utci_misses: bool = False
    _merge_exclude: tuple[str] = ("parameters", "sources")

    @classmethod
    def from_schema(cls, schema_config: dict, **overrides) -> Self:
        paramId = schema_config["metadata"].pop("paramId")
        schema_config["out_params"] = [paramId]

        outputs = overrides.setdefault("outputs", {})
        for out_name in io.ThermoOutputModel.names:
            if out_name != "indices" and out_name not in outputs:
                outputs["out_name"] = {"target": {"type": "null"}}
        return super().from_schema(schema_config, **overrides)

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        for param in self.parameters:
            sources = param.in_sources(self.sources, "accum")
            if any([src.type == "null" for src in sources]):
                for out_req in param.accumulations["step"].out_mars("step"):
                    steps = out_req["step"]
                    if isinstance(steps, (str, int)):
                        steps = [steps]
                    nsteps = list(map(lambda x: len(str(x).split("-")), steps))
                    assert np.all(
                        np.asarray(nsteps) == 1
                    ), f"Accumulation source required for step ranges."
        return self

    @classmethod
    def _populate_sources(
        cls, inputs: list[dict], accum_dims: list[str], **overrides
    ) -> dict:
        sorted_requests = {}
        for inp in inputs:
            if isinstance(inp["step"], list) and len(inp["step"]) > 1:
                src_name = "accum"
            else:
                src_name = "inst"
            [inp.pop(dim, None) for dim in accum_dims]
            sorted_requests.setdefault(src_name, []).append(inp)

        sources = {}
        for src_name, requests in sorted_requests.items():
            src_overrides = overrides.get(src_name, {})
            request_overrides = src_overrides.pop("request", {})
            updated_inputs = update_request(requests, request_overrides)
            sources[src_name] = {
                "request": updated_inputs
                if len(updated_inputs) > 1
                else updated_inputs[0],
                **src_overrides,
            }
        return sources

    def _merge_parameters(self, other: Self) -> list[ThermoParamConfig]:
        merged_params = [self.parameters[0]]
        for in_param in self.parameters[1:] + other.parameters:
            merged = False
            for index, out_param in enumerate(merged_params):
                if out_param.can_merge(in_param):
                    merged_params[index] = out_param.merge(in_param)
                    merged = True
                    break
            if not merged:
                merged_params.append(in_param)
        return merged_params

    def _merge_sources(self, other: Self) -> io.ThermoSourceModel:
        new_sources = self.sources.model_copy()
        other_sources = other.sources.model_copy()
        if new_sources.accum.type_ == "null":
            new_sources.accum.type_ = other_sources.accum.type_
            new_sources.accum.path = other_sources.accum.path
        if other_sources.accum.type_ == "null" and new_sources.accum.type_ != "null":
            other_sources.accum.type_ = new_sources.accum.type_
            other_sources.accum.path = new_sources.accum.path
        if new_sources != other_sources:
            raise ValueError(
                "Can only merge configs with sources differing by accum source type"
            )
        return new_sources
