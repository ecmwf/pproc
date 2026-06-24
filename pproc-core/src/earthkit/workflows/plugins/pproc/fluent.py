# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import inspect
from typing import Optional, Union

import numpy as np
from earthkit.workflows.backends.earthkit import FieldListBackend
from earthkit.workflows.nodetree import nodetree_size

from earthkit.workflows import fluent
from earthkit.workflows.nodetree import nodetree_arrays, nodetree_dimensions
from earthkit.workflows.plugins.pproc.utils.request import MultiSourceRequest
from earthkit.workflows.plugins.pproc.utils.request import Request
from earthkit.workflows.plugins.pproc.utils.metadata import fill_template_values
from earthkit.workflows.plugins.pproc.config.mask import MaskExpression
from earthkit.workflows.plugins.pproc.config.threshold import Threshold
from earthkit.workflows.plugins.pproc.config.accumulation import (
    Coords,
    Default,
    Monthly,
)
from earthkit.workflows.plugins.pproc.metadata.accumulation import accumulation_metadata
from earthkit.workflows.plugins.pproc.metadata.threshold import threshold_metadata


class Action(fluent.Action):
    # TODO: migrate to schema
    _THERMAL_CONFIG = {
        "utci": {
            "operation": "ppruntime.thermal_indices.calc_utci",
            "params": ["2t", "2d", "10si", "mrt"],
        },
        "10si": {
            "operation": "norm",
            "metadata": {"paramId": 207},
            "params": ["10u", "10v"],
        },
        "mrt": {
            "operation": "ppruntime.thermal_indices.calc_mrt",
            "params": ["cossza", "dsrp", "ssrd", "fdir", "strd", "str", "ssr"],
        },
        "cossza": {
            "operation": "ppruntime.thermal_indices.calc_cossza",
            "params": ["2t", "fdir"],
        },
        "dsrp": {
            "operation": "ppruntime.thermal_indices.calc_dsrp",
            "params": ["fdir", "cossza"],
        },
        "hmdx": {
            "operation": "ppruntime.thermal_indices.calc_hmdx",
            "params": ["2t", "2d"],
        },
        "2r": {
            "operation": "ppruntime.thermal_indices.calc_rhp",
            "params": ["2t", "2d"],
        },
        "heatx": {
            "operation": "ppruntime.thermal_indices.calc_heatx",
            "params": ["2t", "2d"],
        },
        "wbgt": {
            "operation": "ppruntime.thermal_indices.calc_wbgt",
            "params": ["2t", "2d", "10si", "mrt"],
        },
        "gt": {
            "operation": "ppruntime.thermal_indices.calc_gt",
            "params": ["2t", "10si", "mrt"],
        },
        "nefft": {
            "operation": "ppruntime.thermal_indices.calc_nefft",
            "params": ["2t", "10si", "2r"],
        },
        "wcf": {
            "operation": "ppruntime.thermal_indices.calc_wcf",
            "params": ["2t", "10si"],
        },
        "aptmp": {
            "operation": "ppruntime.thermal_indices.calc_aptmp",
            "params": ["2t", "2r", "10si"],
        },
    }

    def _reduction_with_metadata(
        self,
        operation: str,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> "Action":
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
    ) -> "Action":
        return self._reduction_with_metadata(
            "sum", dim=dim, batch_size=batch_size, keep_dim=keep_dim, metadata=metadata
        )

    def mean(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> "Action":
        return self._reduction_with_metadata(
            "mean", dim=dim, batch_size=batch_size, keep_dim=keep_dim, metadata=metadata
        )

    def std(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> "Action":
        return self._reduction_with_metadata(
            "std", dim=dim, batch_size=batch_size, keep_dim=keep_dim, metadata=metadata
        )

    def max(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> "Action":
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
    ) -> "Action":
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
    ) -> "Action":
        return self._reduction_with_metadata(
            "prod", dim=dim, batch_size=batch_size, keep_dim=keep_dim, metadata=metadata
        )

    def norm(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> "Action":
        return self._reduction_with_metadata(
            fluent.Payload(FieldListBackend.norm),
            dim=dim,
            batch_size=batch_size,
            keep_dim=keep_dim,
            metadata=metadata,
        )

    def subtract(
        self, other: "Action | float", metadata: dict | None = None
    ) -> "Action":
        return super().subtract(other, backend_kwargs={"metadata": metadata})

    def divide(self, other: "Action | float", metadata: dict | None = None) -> "Action":
        return super().divide(other, backend_kwargs={"metadata": metadata})

    def add(self, other: "Action | float", metadata: dict | None = None) -> "Action":
        return super().add(other, backend_kwargs={"metadata": metadata})

    def multiply(
        self, other: "Action | float", metadata: dict | None = None
    ) -> "Action":
        return super().multiply(other, backend_kwargs={"metadata": metadata})

    scale = multiply

    def power(self, other: "Action | float", metadata: dict | None = None) -> "Action":
        return super().power(other, backend_kwargs={"metadata": metadata})

    def extreme(
        self,
        operation: str,
        climatology: fluent.Action,
        step_ranges: list[str],
        ensemble_dim: str = "number",
        step_dim: str = "step",
        **kwargs,
    ) -> "Action":
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
    ) -> "Action":
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
                {"metadata": metadata},
            )
            return self.join(climatology, "**datatype**").reduce(payload)

        join = self.join(climatology, "**datatype**", match_coord_values=True)
        return join.transform(
            _efi_window_transform,
            [({dim: srange}, eps, metadata) for srange in step_ranges],
            dim,
        )

    def sot(
        self,
        climatology: fluent.Action,
        step_ranges: list[str],
        eps: float,
        sot: list[int],
        dim: str = "step",
        new_dim: str = "sot",
        metadata: dict | None = None,
    ) -> "Action":
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
        return ret

    def threshold_probability(
        self,
        thresholds: list[Union[Threshold, dict]],
        dim: str = "number",
        batch_size: int = 0,
        metadata: Optional[dict] = None,
        clim_metadata: Optional[dict] = None,
    ) -> "Action":
        combined = None
        for threshold in thresholds:
            if isinstance(threshold, dict):
                threshold = Threshold(**threshold)
            selected = self.sel(threshold.select) if threshold.select else self
            selected = selected.map(
                fluent.Payload(
                    "ppruntime.stats.mask",
                    (fluent.Node.input_name(0),),
                    threshold.model_dump(
                        exclude=("select", "lower_scale_factor", "upper_scale_factor")
                    ),
                )
            )
            if combined is None:
                combined = selected
            else:
                combined = combined.join(dim="param").reduce(
                    "ppruntime.stats.logical_and", dim="param"
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
    ) -> "Action":
        anom = self.subtract(clim_mean, metadata=metadata)
        if not std_anomaly:
            return anom
        return anom.divide(clim_std)

    def quantiles(
        self,
        quantiles: int | list[float] | str,
        dim: str = "number",
        new_dim: str = "quantile",
        metadata: dict | None = None,
    ) -> "Action":
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
        return ret

    def _wrapped_reduction(
        self,
        operation: str | fluent.Payload | None,
        dim: str,
        batch_size: int = 0,
        metadata: dict | None = None,
        **kwargs,
    ) -> "Action":
        if operation is None:
            self._squeeze_dimension(dim)
            if metadata is None:
                return self
            return self.map(
                fluent.Payload(
                    FieldListBackend.set_metadata, [fluent.Node.input_name(0), metadata]
                )
            )
        if isinstance(operation, str):
            if hasattr(self, operation):
                op = getattr(self, operation)
                sig = inspect.signature(op)
                args = [
                    p.name
                    for p in sig.parameters.values()
                    if p.kind == p.POSITIONAL_OR_KEYWORD
                ]
                if "batch_size" in args:
                    kwargs["batch_size"] = batch_size
                if "dim" in args:
                    kwargs["dim"] = dim
                return op(metadata=metadata, **kwargs)
            if metadata is not None:
                kwargs.setdefault("metadata", {}).update(metadata)
            op_function = getattr(FieldListBackend, operation, None) or operation
            operation = fluent.Payload(
                op_function,
                kwargs=kwargs,
            )
        return self.reduce(operation, dim=dim, batch_size=batch_size)

    def preprocessing(
        self,
        operation: str | fluent.Payload | None,
        dim: str = "param",
        batch_size: int = 0,
        metadata: dict | None = None,
        **kwargs,
    ) -> "Action":
        """
        Reduction operation across different parameters

        Params
        ------
        operation: str or Payload, operation to perform on ensemble members
        dim: str, dimension to perform operation along

        Return
        ------
        Action
        """
        return self._wrapped_reduction(operation, dim, batch_size, metadata, **kwargs)

    # TODO: turn this into more general param operation that scans schema to look for
    # computation method of missing parameters. Or should it be that in the creation of the
    # configuration returned by the schema, we list configuration for creating all required
    # intermediate parameters
    def thermal_index(
        self, param: str, dim: str = "param", metadata: dict | None = None
    ) -> "Action":
        config = self._THERMAL_CONFIG[param]
        new_action = self
        for inp in config["params"]:
            if (
                inp not in new_action.nodes.coords[dim]
                and inp in new_action._THERMAL_CONFIG
            ):
                dependency = new_action.thermal_index(inp, dim=dim, metadata=metadata)
                dependency._add_dimension(dim, inp)
                new_action = new_action.join(dependency, dim)
        try:
            selection = new_action.sel(param=(config["params"]))
        except KeyError:
            selection = new_action
        config_metadata = config.get("metadata", {})
        new_metadata = (
            config_metadata if not metadata else {**metadata, **config_metadata}
        )
        ret = selection._wrapped_reduction(
            config["operation"], dim=dim, metadata=new_metadata
        )
        return ret

    def mask(
        self,
        select: dict,
        mask: Union[MaskExpression, dict],
        replacement: float = 0.0,
        dim: str = "param",
        metadata: dict | None = None,
    ) -> "Action":
        if isinstance(mask, dict):
            mask = MaskExpression(**mask)
        mask_action = self.sel(mask.select, drop=True) if select else self
        mask_action = mask_action.map(
            fluent.Payload(
                "ppruntime.stats.mask",
                (fluent.Node.input_name(0),),
                {"lower_comparison": mask.comparison, "lower_value": mask.value},
            )
        )
        return (
            self.sel(select, drop=True)
            .join(mask_action, dim=dim)
            ._wrapped_reduction(
                FieldListBackend.filter,
                dim=dim,
                metadata=metadata,
                replacement=replacement,
            )
        )

    def ensemble_statistics(
        self,
        operation: str | fluent.Payload | None,
        dim: str = "number",
        batch_size: int = 0,
        metadata: dict | None = None,
        **kwargs,
    ) -> "Action":
        """
        Reduction operation across ensemble members. If batch_size > 1 and less than the size
        of the named dimension, the reduction will be computed first in
        batches and then aggregated, otherwise no batching will be performed.

        Params
        ------
        operation: str or Payload, operation to perform on ensemble members
        dim: str, dimension to perform operation along
        batch_size: int, size of batches to split reduction into. If 0,
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

        return self._wrapped_reduction(
            operation, dim, batch_size, stat_metadata, **kwargs
        )

    def accumulation(
        self,
        operation: Optional[str | fluent.Payload],
        coords: list[Coords],
        dim: str = "step",
        batch_size: int = 0,
        metadata: dict | None = None,
        deaccumulate: bool = False,
        name: Union[Default, Monthly, dict] = Default(),
        **kwargs,
    ) -> "Action":
        """
        Reduction operation across a dimension. If batch_size > 1 and less than the size
        of the named dimension, the reduction will be computed first in
        batches and then aggregated, otherwise no batching will be performed.

        Params
        ------
        operation: str or Payload, operation to perform on steps
        coords: list of values to accumulate over
        dim: str, dimension to perform operation along
        batch_size: int, size of batches to split reduction into. If 0,
        computation is not batched
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
                batch_size,
                metadata,
                deaccumulate,
                name,
                kwargs,
            )
            for coord in coords
        ]
        return self.transform(_accum_transform, params, dim)

    def write(self, targets: list[dict], metadata: Optional[dict] = None) -> "Action":
        if len(targets) == 0:
            raise ValueError("No targets provided for write")
        return self.transform(
            _write_transform,
            [(target, metadata) for target in targets],
            dim="target",
        )


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
        )
    )


def _sot_transform(
    action: fluent.Action, number: int, eps: float, new_dim: str, metadata: dict | None
) -> fluent.Action:
    new_sot = action.reduce(
        fluent.Payload(
            "ppruntime.stats.sot",
            (fluent.Node.input_name(1), fluent.Node.input_name(0), number, eps),
            {"metadata": metadata},
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
            {"metadata": metadata},
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
        {"metadata": metadata},
    )
    new_quantile = action.map(payload)
    new_quantile._add_dimension(new_dim, q_number / total_number)
    return new_quantile


def _accum_transform(
    action: fluent.Action,
    dim: str,
    coords: Coords,
    operation: str | fluent.Payload,
    batch_size: int,
    metadata: Optional[dict] = None,
    deaccumulate: bool = False,
    name: Union[Default, Monthly, dict] = Default(),
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
        name = Default(**name) if name.get("type_") == "default" else Monthly(**name)
    accum_name = name.name(coords)
    accum_metadata = accumulation_metadata(dim, coords, accum_name, metadata)

    accum_action = accum_action._wrapped_reduction(
        operation,
        dim=dim,
        batch_size=batch_size,
        metadata=accum_metadata,
        **kwargs,
    )

    accum_action._add_dimension(dim, accum_name)

    return accum_action


def from_source(
    sources: list[Union[str, dict]],
    requests: list[dict | Request | MultiSourceRequest],
    dtype: Optional[str] = None,
    join_key: str = "",
):
    all_actions = None
    for request in requests:
        if isinstance(request, dict):
            request = Request(request)
        payloads = np.empty(tuple(request.dims.values()), dtype=object)
        for indices, new_request in request.expand():
            payloads[indices] = fluent.Payload(
                "ppruntime.io.retrieve", [sources, [new_request], dtype]
            )
        new_action = fluent.from_source(
            payloads,
            coords={key: list(request[key]) for key in request.dims.keys()},
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
