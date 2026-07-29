# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import inspect
from typing import Optional, Union, Any
from pydantic import TypeAdapter
import logging

import numpy as np
from earthkit.workflows.backends.earthkit import FieldListBackend
from earthkit.workflows.nodetree import nodetree_size

from earthkit.workflows import fluent
from earthkit.workflows.nodetree import (
    nodetree_array,
    nodetree_arrays,
    nodetree_dimensions,
    nodetree_from_dict,
)
from earthkit.workflows.plugins.pproc.utils.request import MultiSourceRequest
from earthkit.workflows.plugins.pproc.utils.request import Request
from earthkit.workflows.plugins.pproc.utils.metadata import fill_template_values
from earthkit.workflows.plugins.pproc.config.mask import MaskExpression
from earthkit.workflows.plugins.pproc.config.threshold import Threshold
from earthkit.workflows.plugins.pproc.config.accumulation import (
    Coords,
    Default,
    AccumName,
)
from earthkit.workflows.plugins.pproc.metadata.accumulation import accumulation_metadata
from earthkit.workflows.plugins.pproc.metadata.threshold import threshold_metadata

# TODO: change git url to ppruntime when published in PyPI
ENVIRONMENT = {
    "ppruntime": [
        "pproc-runtime @ git+https://git@github.com/ecmwf/pproc.git@feature/mono-repo#subdirectory=pproc-runtime"
    ],
}


logger = logging.getLogger(__name__)


class Action(fluent.Action):
    def _reduction_with_metadata(
        self,
        operation: str,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> Action:
        batched = batch_size > 1 and self.nodes.sizes[dim] > batch_size

        if not batched or metadata is None:
            if isinstance(operation, str):
                if hasattr(super(), operation):
                    return getattr(super(), operation)(
                        dim=dim,
                        batch_size=batch_size,
                        keep_dim=keep_dim,
                        backend_kwargs={"metadata": metadata},
                    )
                else:
                    operation = (
                        fluent.Payload(operation, kwargs={"metadata": metadata}),
                    )
            else:
                operation.kwargs.setdefault("metadata", {}).update(metadata or {})
            return super().reduce(
                operation,
                dim=dim,
                batch_size=batch_size,
                keep_dim=keep_dim,
            )

        if isinstance(operation, str):
            if hasattr(super(), operation):
                batched_action = getattr(super(), operation)(
                    dim=dim, batch_size=batch_size, keep_dim=keep_dim
                )
            else:
                batched_action = super().reduce(
                    fluent.Payload(operation),
                    dim=dim,
                    batch_size=batch_size,
                    keep_dim=keep_dim,
                )
        else:
            batched_action = super().reduce(
                operation, dim=dim, batch_size=batch_size, keep_dim=keep_dim
            )
        # If batched, add additional node for setting window operation metadata. Doing this in a separate tasks
        # allows batched operations for overlapping windows to be identified and only computed once
        return batched_action.map(
            fluent.Payload(
                FieldListBackend.set_metadata, [fluent.Node.input_name(0), metadata]
            )
        )

    def sum(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> Action:
        return self._reduction_with_metadata(
            "sum", dim=dim, batch_size=batch_size, keep_dim=keep_dim, metadata=metadata
        )

    def mean(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> Action:
        return self._reduction_with_metadata(
            "mean", dim=dim, batch_size=batch_size, keep_dim=keep_dim, metadata=metadata
        )

    def std(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> Action:
        return self._reduction_with_metadata(
            "std", dim=dim, batch_size=batch_size, keep_dim=keep_dim, metadata=metadata
        )

    def max(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> Action:
        return self._reduction_with_metadata(
            "max", dim=dim, batch_size=batch_size, keep_dim=keep_dim, metadata=metadata
        )

    maximum = max

    def min(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> Action:
        return self._reduction_with_metadata(
            "min", dim=dim, batch_size=batch_size, keep_dim=keep_dim, metadata=metadata
        )

    minimum = min

    def prod(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> Action:
        return self._reduction_with_metadata(
            "prod", dim=dim, batch_size=batch_size, keep_dim=keep_dim, metadata=metadata
        )

    def norm(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> Action:
        return self._reduction_with_metadata(
            fluent.Payload(FieldListBackend.norm),
            dim=dim,
            batch_size=batch_size,
            keep_dim=keep_dim,
            metadata=metadata,
        )

    def subtract(
        self, other: Union[Action, float], metadata: dict | None = None
    ) -> Action:
        return super().subtract(other, backend_kwargs={"metadata": metadata})

    def divide(
        self, other: Union[Action, float], metadata: dict | None = None
    ) -> Action:
        return super().divide(other, backend_kwargs={"metadata": metadata})

    def add(self, other: Union[Action, float], metadata: dict | None = None) -> Action:
        return super().add(other, backend_kwargs={"metadata": metadata})

    def multiply(
        self, other: Union[Action, float], metadata: dict | None = None
    ) -> Action:
        return super().multiply(other, backend_kwargs={"metadata": metadata})

    scale = multiply

    def power(
        self, other: Union[Action, float], metadata: dict | None = None
    ) -> Action:
        return super().power(other, backend_kwargs={"metadata": metadata})

    def extreme(
        self,
        operation: str,
        climatology: fluent.Action,
        step_ranges: list[str],
        ensemble_dim: str = "number",
        step_dim: str = "step",
        **kwargs,
    ) -> Action:
        return self.concatenate(ensemble_dim).__getattribute__(operation)(
            climatology, step_ranges, dim=step_dim, **kwargs
        )

    def efi(
        self,
        climatology: fluent.Action,
        step_ranges: list[str],
        eps: float,
        dim: str = "step",
        metadata: dict | None = None,
    ) -> Action:
        """
        Create nodes computing the EFI for each window. Expects ensemble member dimension
        to already be concatenated into a single array.

        Parameters
        ----------
        climatology: Action, nodes containing climatology data
        step_ranges: list of str, list of step ranges
        eps: float
        dim: str, window dimension

        Return
        ------
        Action
        """
        eps = float(eps)
        if nodetree_size(self.nodes) == 1:
            if len(step_ranges) != 1:
                raise ValueError("Single node, but multiple step ranges")
            payload = fluent.Payload(
                "ppruntime.stats.efi",
                (fluent.Node.input_name(1), fluent.Node.input_name(0), eps),
                kwargs={"metadata": metadata},
                metadata={"environment": ENVIRONMENT["ppruntime"]},
            )
            return self.join(climatology, "**datatype**").reduce(payload)  # type: ignore

        join = self.join(climatology, "**datatype**", match_coord_values=True)
        return join.transform(
            _efi_window_transform,
            [({dim: srange}, eps, metadata) for srange in step_ranges],
            dim,
        )  # type: ignore

    def sot(
        self,
        climatology: fluent.Action,
        step_ranges: list[str],
        eps: float,
        sot: list[int],
        dim: str = "step",
        new_dim: str = "sot",
        metadata: dict | None = None,
    ) -> Action:
        """
        Create nodes computing the SOT for each window. Expects ensemble member dimension
        to already be concatenated into a single array.

        Parameters
        ----------
        climatology: Action, nodes containing climatology data
        step_ranges: list of str, list of step ranges
        eps: float
        sot: list of ints, Shift-Of-Tail values
        dim: str, window dimension
        new_dim: str, name of new dimension corresponding to SOT nodes.

        Return
        ------
        Action
        """
        eps = float(eps)
        if not isinstance(sot, list):
            sot = [sot]

        if nodetree_size(self.nodes) == 1:
            if len(step_ranges) != 1:
                raise ValueError("Single node, but multiple step ranges")
            ret = self.join(
                climatology, "**datatype**", match_coord_values=True
            ).transform(
                _sot_transform,
                [(int(num), eps, new_dim, metadata) for num in sot],
                new_dim,
            )
        else:
            params = [
                ({dim: srange}, sot, eps, new_dim, metadata) for srange in step_ranges
            ]
            ret = self.join(
                climatology, "**datatype**", match_coord_values=True
            ).transform(_sot_window_transform, params, dim)
        return ret  # type: ignore

    def threshold_probability(
        self,
        thresholds: list[Union[Threshold, dict]],
        dim: str = "number",
        batch_size: int = 0,
        metadata: Optional[dict] = None,
        clim_metadata: Optional[dict] = None,
    ) -> Action:
        combined: Action = None
        for threshold in thresholds:
            if isinstance(threshold, dict):
                threshold = Threshold(**threshold)
            selected = self.sel(threshold.select) if threshold.select else self
            selected = selected.map(
                fluent.Payload(
                    "ppruntime.stats.mask",
                    (fluent.Node.input_name(0),),
                    threshold.model_dump(
                        exclude={"select", "lower_scale_factor", "upper_scale_factor"}
                    ),
                    metadata={"environment": ENVIRONMENT["ppruntime"]},
                )
            )
            if combined is None:
                combined = selected
            else:
                combined = combined.join(selected, dim="param").reduce(  # type: ignore
                    fluent.Payload(
                        "ppruntime.stats.logical_and",
                        metadata={"environment": ENVIRONMENT["ppruntime"]},
                    ),
                    dim="param",
                )

        base_threshold = (
            thresholds[0]
            if isinstance(thresholds[0], Threshold)
            else Threshold(**thresholds[0])
        )
        thr_metadata = threshold_metadata(
            threshold=base_threshold, metadata=metadata, clim_metadata=clim_metadata
        )
        return combined.multiply(100).mean(
            dim=dim, batch_size=batch_size, metadata=thr_metadata
        )

    def anomaly(
        self,
        clim_mean: fluent.Action,
        clim_std: fluent.Action,
        std_anomaly: bool = False,
        metadata: dict | None = None,
    ) -> Action:
        anom = self.subtract(clim_mean, metadata=metadata)  # type: ignore
        if not std_anomaly:
            return anom
        return anom.divide(clim_std)  # type: ignore

    def quantiles(
        self,
        quantiles: int | list[float] | str,
        dim: str = "number",
        new_dim: str = "quantile",
        metadata: dict | None = None,
    ) -> Action:
        """
        Compute quantiles over the specified dimension. If quantiles is an integer, it will compute that many evenly spaced quantiles.
        If quantiles is a string, it should be in the format "q_number:total_number".
        If quantiles is a list of floats, it will compute the quantiles at the specified values.

        Parameters
        ----------
        quantiles: int | list[float] | str, quantiles to compute
        dim: str, dimension to compute quantiles over
        new_dim: str, name of the new dimension for quantiles
        metadata: dict, optional metadata to set on the output
        """
        if isinstance(quantiles, int):
            total_number = quantiles
            q_numbers = range(0, quantiles + 1)
        elif isinstance(quantiles, str):
            q_number, total_number = map(int, quantiles.split(":"))
            q_numbers = [q_number]
        else:
            even_spacing = np.all(np.diff(quantiles) == quantiles[1] - quantiles[0])
            total_number = len(quantiles) - 1 if even_spacing else 100
            q_numbers = (
                range(0, len(quantiles)) if even_spacing else map(int, quantiles * 100)
            )

        params = [(x, total_number, new_dim, metadata) for x in q_numbers]
        ret = self.concatenate(dim).transform(_quantiles_transform, params, new_dim)
        return ret  # type: ignore

    def _wrapped_reduction(
        self,
        operation: str | fluent.Payload | None,
        dim: str,
        **kwargs,
    ) -> Action:
        if operation is None:
            return self

        if isinstance(operation, str):
            # operation is fluent method
            if hasattr(self, operation):
                op = getattr(self, operation)
                sig = inspect.signature(op)
                args = [
                    p.name
                    for p in sig.parameters.values()
                    if p.kind == p.POSITIONAL_OR_KEYWORD
                ]
                if "dim" in args:
                    kwargs["dim"] = dim
                return op(**kwargs)
            # operation is payload
            op_function = getattr(FieldListBackend, operation, None) or operation
            operation = fluent.Payload(
                op_function,
            )
        # Fallback on applying reduction operation during payload
        return self.reduce(operation, dim=dim, **kwargs)  # type: ignore

    def preprocessing(
        self,
        operation: str | fluent.Payload | None,
        dim: str = "param",
        **kwargs,
    ) -> Action:
        """
        Pre-processing operation across different parameters

        Params
        ------
        operation: str or Payload, operation to perform on ensemble members
        dim: str, dimension to perform operation along

        Return
        ------
        Action
        """
        return self._wrapped_reduction(operation, dim, **kwargs)

    def thermal_index(
        self,
        function: str,
        params: list[str],
        deaccumulate: Optional[list[str]] = None,
        dim: str = "param",
        join: bool = True,
        metadata: dict | None = None,
    ) -> Action:
        """
        Thermal index computation

        Params
        ------
        function: str, pproc-runtime function to call for thermal index computation
        params: list[str], list of input parameters
        deaccumulate: list[str], list of parameters that require deaccumulating
        dim: str, parameter dimension
        join: bool, whether to join the resulting action with input action
        metadata: dict, metadata to set on the output
        """
        logger.debug(
            f"Thermal index: function {function}, params {params}, deaccumulate {deaccumulate}"
        )
        if deaccumulate is not None:
            deaccum = self.sel(param=deaccumulate)
            array = nodetree_array(deaccum.nodes)
            steps = array.coords["step"].data.tolist()
            deaccum = deaccum.accumulation(
                operation=None,
                coords=[[steps[x], steps[x + 1]] for x in range(len(steps) - 1)],
                deaccumulate=True,
                metadata={"stepType": "diff"},
            )
            deaccum = type(self)(
                nodetree_from_dict(
                    {
                        path: array.assign_coords(
                            {"step": steps[1] if len(steps) <= 2 else steps[1:]}
                        )
                        for path, array in nodetree_arrays(deaccum.nodes)
                    }
                )
            )
            param_action = fluent.merge(
                self.select(
                    param=[x for x in params if x not in deaccumulate],
                    step=steps[1:],
                    expand=True,
                ),
                deaccum,
            )
            param_action._squeeze_dimension("step")
            param_action = param_action.combine_branches(dim=dim)
        else:
            param_action = self.select(param=params, expand=True)
            param_action._squeeze_dimension("step")
            param_action = param_action.combine_branches(dim=dim)

        if method := getattr(param_action, function, None):
            ret = method(dim=dim, metadata=metadata)
        else:
            ret = param_action._wrapped_reduction(
                fluent.Payload(func=function, kwargs={"metadata": metadata or {}}),
                dim=dim,
            )
        ret._add_dimension(dim, str(metadata.get("paramId")))
        if not join:
            return ret
        ret = ret.set_path(f"/{metadata.get('paramId')}")
        return fluent.merge(self, ret)

    def mask(
        self,
        select: dict,
        mask: Union[MaskExpression, dict],
        replacement: float = 0.0,
        dim: str = "param",
        metadata: dict | None = None,
    ) -> Action:
        if isinstance(mask, dict):
            mask = MaskExpression(**mask)
        mask_action = self.sel(mask.select, drop=True) if select else self
        mask_action = mask_action.map(
            fluent.Payload(
                "ppruntime.stats.mask",
                (fluent.Node.input_name(0),),
                {"lower_comparison": mask.comparison, "lower_value": mask.value},
                metadata={"environment": ENVIRONMENT["ppruntime"]},
            )
        )
        return (
            self.sel(select, drop=True)
            .join(mask_action, dim=dim)
            ._wrapped_reduction(
                fluent.Payload(
                    FieldListBackend.filter, kwargs={"replacement": replacement}
                ),
                dim=dim,
                metadata=metadata,
            )
        )

    def ensemble_statistics(
        self,
        operation: str | fluent.Payload | None,
        dim: str = "number",
        metadata: dict | None = None,
        **kwargs,
    ) -> Action:
        """
        Reduction operation across ensemble members. If batch_size > 1 and less than the size
        of the named dimension, the reduction will be computed first in
        batches and then aggregated, otherwise no batching will be performed.

        Params
        ------
        operation: str or Payload, operation to perform on ensemble members
        dim: str, dimension to perform operation along
        computation is not batched

        Return
        ------
        Action

        Raises
        ------
        ValueError if payload function is not batchable and batch_size is not 0
        """
        metadata = metadata or {}

        for _, narray in nodetree_arrays(self.nodes):
            if dim not in narray.coords:
                raise ValueError(
                    f"Dimension {dim} not found in nodes for ensemble_statistics"
                )
            stat_metadata = fill_template_values(
                metadata, {"num_fields": narray.coords[dim].size}
            )
            break

        kwargs.setdefault("metadata", {}).update(stat_metadata)
        return self._wrapped_reduction(operation, dim, **kwargs)

    def accumulation(
        self,
        operation: Optional[str | fluent.Payload],
        coords: list[Coords],
        dim: str = "step",
        metadata: dict | None = None,
        deaccumulate: bool = False,
        name: Union[AccumName, dict] = Default(),
        **kwargs,
    ) -> Action:
        """
        Reduction operation across a dimension. If batch_size > 1 and less than the size
        of the named dimension, the reduction will be computed first in
        batches and then aggregated, otherwise no batching will be performed.

        Params
        ------
        operation: str or Payload, operation to perform on steps
        coords: list of values to accumulate over
        dim: str, dimension to perform operation along
        metadata: optional dict, metadata to set on the output
        deaccumulate: bool, whether to deaccumulate consecutive values before accumulation

        Return
        ------
        Action

        Raises
        ------
        ValueError if payload function is not batchable and batch_size is not 0
        """
        params = [
            (
                dim,
                coord,
                operation,
                metadata,
                deaccumulate,
                name,
                kwargs,
            )
            for coord in coords
        ]
        return self.transform(_accum_transform, params, dim)  # type: ignore

    def write(self, targets: list[dict], metadata: Optional[dict] = None) -> Action:
        if len(targets) == 0:
            raise ValueError("No targets provided for write")
        return self.transform(
            _write_transform,
            [(target, metadata) for target in targets],
            dim="target",
        )  # type: ignore


def _write_transform(
    action: fluent.Action, target: dict, metadata: Optional[dict] = None
) -> fluent.Action:
    kwargs = target.copy()
    if metadata is not None:
        kwargs["metadata"] = metadata
    return action.map(
        fluent.Payload(
            "ppruntime.io.write",
            (fluent.Node.input_name(0),),
            kwargs,
            metadata={"environment": ENVIRONMENT["ppruntime"]},
        )
    )


def _sot_transform(
    action: fluent.Action, number: int, eps: float, new_dim: str, metadata: dict | None
) -> fluent.Action:
    new_sot = action.reduce(
        fluent.Payload(
            "ppruntime.stats.sot",
            (fluent.Node.input_name(1), fluent.Node.input_name(0), number, eps),
            kwargs={"metadata": metadata},
            metadata={"environment": ENVIRONMENT["ppruntime"]},
        )
    )
    new_sot._add_dimension(new_dim, number)
    return new_sot


def _sot_window_transform(
    action: fluent.Action,
    selection: dict,
    sot: list[int],
    eps: float,
    new_dim: str,
    metadata: dict,
) -> fluent.Action:
    return action.select(selection).transform(
        _sot_transform, [(int(num), eps, new_dim, metadata) for num in sot], new_dim
    )


def _efi_window_transform(
    action: fluent.Action, selection: dict, eps: float, metadata: dict | None
) -> fluent.Action:
    ret = action.select(selection).reduce(
        fluent.Payload(
            "ppruntime.stats.efi",
            (fluent.Node.input_name(1), fluent.Node.input_name(0), eps),
            kwargs={"metadata": metadata},
            metadata={"environment": ENVIRONMENT["ppruntime"]},
        ),
        dim="**datatype**",
    )
    return ret


def _quantiles_transform(
    action, q_number: int, total_number: int, new_dim: str, metadata: dict | None
):
    payload = fluent.Payload(
        "ppruntime.stats.quantiles",
        (fluent.Node.input_name(0), q_number, total_number),
        kwargs={"metadata": metadata},
        metadata={"environment": ENVIRONMENT["ppruntime"]},
    )
    new_quantile = action.map(payload)
    new_quantile._add_dimension(new_dim, q_number / total_number)
    return new_quantile


def _accum_transform(
    action: fluent.Action,
    dim: str,
    coords: Coords,
    operation: str | fluent.Payload,
    metadata: Optional[dict] = None,
    deaccumulate: bool = False,
    name: Union[AccumName, dict] = Default(),
    kwargs: dict = {},
) -> fluent.Action:
    if deaccumulate:
        accum_action = action.select({dim: coords[:-1]})
        accum_action = accum_action.subtract(action.select({dim: coords[1:]}))
    else:
        if len(coords) == 1 and dim not in nodetree_dimensions(action.nodes):
            accum_action = action
        else:
            accum_action = action.select({dim: coords})

    if isinstance(name, dict):
        name = TypeAdapter(AccumName, name).validate_python()
    accum_name = name.name(coords)
    accum_metadata = accumulation_metadata(dim, coords, accum_name, metadata)

    if operation is None:
        accum_action._squeeze_dimension(dim)
        if len(accum_metadata) > 0:
            accum_action = accum_action.map(
                fluent.Payload(
                    FieldListBackend.set_metadata,
                    [fluent.Node.input_name(0), accum_metadata],
                )
            )
    else:
        accum_action = accum_action._wrapped_reduction(
            operation,
            dim=dim,
            metadata=accum_metadata,
            **kwargs,
        )

    accum_action._add_dimension(dim, accum_name)

    return accum_action


def from_source(
    sources: list[Union[str, dict]],
    requests: Union[list[dict[str, Any]], list[Request], list[MultiSourceRequest]],
    dtype: Optional[str] = None,
    join_key: str = "",
):
    all_actions = None
    for request in requests:
        if isinstance(request, dict):
            request = Request(request)
        payloads = np.empty(tuple(request.dims().values()), dtype=object)
        for indices, new_request in request.expand():
            payloads[indices] = fluent.Payload(
                "ppruntime.io.retrieve",
                [sources, [new_request], dtype],
                metadata={"environment": ENVIRONMENT["ppruntime"]},
            )
        new_action = fluent.from_source(
            payloads,
            dims=request.dims().keys(),
            coords={
                key: request[key] for key in request.dims(exclude_scalar=False).keys()
            },
            action=Action,
        )

        if len(join_key) != 0 and join_key not in new_action.nodes.coords:
            new_action._add_dimension(join_key, request[join_key])

        if all_actions is None:
            all_actions = new_action
        else:
            if len(join_key) == 0:
                raise ValueError("Join key must be specified for multiple requests")
            all_actions = all_actions.join(new_action, join_key)
    return all_actions


def path_from_request(request: dict, keys: list[str] = ["levtype", "param"]) -> str:
    return "/".join([str(request[x]) for x in keys])
