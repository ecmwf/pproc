# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import functools
import signal
import sys

from conflator import Conflator
from earthkit.data import FieldList
from meters import ResourceMeter

from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.parallel import parallel_processing, sigterm_handler
from pproc.common.param_requester import ParamRequester
from pproc.config.types import ClimParamConfig, RegimesConfig
from pproc.common.utils import dict_product
from pproc.regimes import get_classifier
from pproc.signi.clim import retrieve_clim


def regime_iteration(
    config: RegimesConfig,
    pconfig: ClimParamConfig,
    dims: dict,
):
    ids = ", ".join(f"{k}={v}" for k, v in dims.items())
    target = config.outputs.regimes.target

    with ResourceMeter(f"Load {config.classifier.regimes} regime classifier"):
        classifier = get_classifier(config.classifier.regimes, config.classifier.path)

    total = pconfig.compute_totalfields(config.inputs, "fc")
    requester = ParamRequester(pconfig, config.inputs, total, "fc")
    with ResourceMeter(f"Retrieve fc {ids}"):
        metadata, data = requester.retrieve_data(**dims)

    with ResourceMeter(f"Compute anomalies {ids}"):
        # Each iteration must only cover a single step and level
        steps = {md.get("stepRange") for md in metadata}
        assert len(steps) == 1, f"Accumulations must be split by step, got {steps}"
        additional_dims = {"step": steps.pop()}
        if metadata[0].get("levtype") == "pl":
            levels = {md.get("level") for md in metadata}
            assert len(levels) == 1, (
                f"Accumulations must be split by level, got {levels}"
            )
            additional_dims["levelist"] = levels.pop()
        clim_accum, _ = retrieve_clim(
            pconfig.clim,
            config.inputs,
            "clim",
            pconfig.clim.total_fields,
            **additional_dims,
        )
        assert clim_accum.values is not None
        data -= clim_accum.values.mean(axis=0)

        anomalies = (
            FieldList.from_array(data, [x.to_ekmetadata() for x in metadata])
            .to_xarray(**config.to_xarray_kwargs, add_earthkit_attrs=False)["z"]
            .pipe(classifier.crop_region)
        )

    with ResourceMeter(f"Run classifier"):
        target.write(classifier.classify(anomalies, **config.classifier.project_kwargs))

    target.flush()
    config.recovery.add_checkpoint(param=pconfig.name, **dims)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-regimes", model=RegimesConfig).load()
    cfg.initialise()
    cfg.print()

    plan = []
    for param in cfg.parameters:
        accum_manager = AccumulationManager.create(
            param.accumulations,
        )
        for dims in dict_product(accum_manager.dims):
            if cfg.recovery.existing_checkpoint(param=param.name, **dims):
                print(f"Recovery: skipping dims: {param.name} {dims}")
                continue
            plan.append((param, dims))

        iteration = functools.partial(regime_iteration, cfg)
        parallel_processing(
            iteration,
            plan,
            cfg.parallelisation,
        )

        cfg.clean()


if __name__ == "__main__":
    sys.exit(main())
