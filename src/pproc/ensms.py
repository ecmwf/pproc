#!/usr/bin/env python3
#
# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.
import functools
import sys
from datetime import datetime
import signal
from meters import ResourceMeter
import numpy as np
from typing import Union, Optional, Any
from typing_extensions import Self
from pydantic import (
    Field,
    BeforeValidator,
    ConfigDict,
    field_validator,
    model_validator,
    create_model,
    computed_field,
)
import os

import eccodes
import pyfdb
from conflator import CLIArg, ConfigModel, Conflator
from annotated_types import Annotated

from pproc import common
from pproc.common import parallel
from pproc.common.accumulation import Accumulator
from pproc.common.parallel import (
    parallel_processing,
    sigterm_handler,
    SynchronousExecutor,
    QueueingExecutor,
    parallel_data_retrieval,
)
from pproc.common.param_requester import ParamConfig, ParamRequester


def template_ensemble(
    param_type: ParamConfig,
    template: eccodes.GRIBMessage,
    accum: Accumulator,
    out_keys: dict,
):
    template_ens = template.copy()

    grib_sets = accum.grib_keys().copy()
    grib_sets.update(out_keys)
    template_ens.set(grib_sets)
    return template_ens


class SourceConfig(ConfigModel):
    type_: str = Field(alias="type")
    request: dict | list[dict]
    path: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": "file", "path": data, "request": {}}
        return data


def create_source_model(entrypoint: str, sources: list[str]):
    return create_model(
        f"{entrypoint.capitalize()}Sources",
        **{
            source: (
                Annotated[
                    SourceConfig,
                    CLIArg(f"--in-{source}"),
                    Field(description=f"Input {source} grib file"),
                ],
                ...,
            )
            for source in sources
        },
        __base__=ConfigModel,
    )


def create_target_model(entrypoint: str, targets: list[str]):
    class TargetBase(ConfigModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

    return create_model(
        f"{entrypoint.capitalize()}Targets",
        **{
            target: (
                Annotated[
                    common.io.Target,
                    CLIArg(f"--out-{target}"),  # Not currently supported by Conflator
                    Field(description=f"Output target for {target}"),
                ],
                ...,
            )
            for target in targets
        },
        __base__=TargetBase,
    )


class EnsmsConfig(ConfigModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    members: int | create_model("Members", start=(int, ...), end=(int, ...))
    total_fields: Annotated[int, Field(validate_default=True)] = 0
    date: datetime
    root_dir: str = os.getcwd()
    sources: create_source_model("ensms", ["ens"])
    n_par_read: int = 1
    n_par_compute: int = 1
    queue_size: int = 1
    out_keys: dict = {}
    out_keys_em: dict = {"type": "em"}
    out_keys_es: dict = {"type": "es"}
    steps: list = []
    windows: list = []
    parameters: dict
    override_input: Annotated[
        dict,
        BeforeValidator(lambda x: common.config.parse_var_strs(x)),
        CLIArg("--override-input", action="append", default=[], metavar="KEY=VALUE,...",),
        Field(description="Override input requests with these keys"),
    ] 
    override_output: Annotated[
        dict,
        BeforeValidator(lambda x: common.config.parse_var_strs(x)),
        CLIArg("--override-output", action="append", default=[], metavar="KEY=VALUE,...",),
        Field(description="Override outputs with these keys"),
    ] 
    recover: Annotated[
        bool,
        CLIArg("--recover", action="store_true", default=False),
    ]
    targets: create_target_model("ensms", ["mean", "std"])
    _fdb: Optional[pyfdb.FDB] = None

    @model_validator(mode="after")
    def check_totalfields(self) -> Self:
        if self.total_fields == 0:
            self.total_fields = (
                self.members
                if isinstance(self.members, int)
                else self.members.end - self.members.start + 1
            )
        return self

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.param_configs = [
        #     ParamConfig(pname, popt, overrides=self.override_input)
        #     for pname, popt in self.parameters.items()
        # ]

        for target in [self.targets.mean, self.targets.std]:
            if self.n_par_compute > 1:
                target.enable_parallel(parallel)
            if self.recover:
                target.enable_recovery()

    @model_validator(mode="before")
    def validate_targets(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        targets = data.get("targets", {})
        if not isinstance(targets, dict):
            return data

        for name in targets.keys():
            targets[name] = common.io.target_from_location(
                targets[name], overrides=data["override_output"]
            )
        return data


def ensms_iteration(
    config: EnsmsConfig,
    param: ParamConfig,
    recovery: common.Recovery,
    window_id: str,
    accum: Accumulator,
    template_ens=Union[str, eccodes.GRIBMessage],
):
    if not isinstance(template_ens, eccodes.GRIBMessage):
        template_ens = common.io.read_template(template_ens)

    ens = accum.values
    assert ens is not None

    # Compute mean/std over all dimensions except last
    axes = tuple(range(ens.ndim - 1))
    with ResourceMeter(f"Window {window_id}: write mean output"):
        mean = np.mean(ens, axis=axes)
        template_mean = template_ensemble(
            param, template_ens, accum, config.out_keys_em
        )
        template_mean.set_array("values", common.io.nan_to_missing(template_mean, mean))
        config.out_mean.write(template_mean)
        config.out_mean.flush()

    with ResourceMeter(f"Window {window_id}: write std output"):
        std = np.std(ens, axis=axes)
        template_std = template_ensemble(param, template_ens, accum, config.out_keys_es)
        template_std.set_array("values", common.io.nan_to_missing(template_std, std))
        config.out_std.write(template_std)
        config.out_std.flush()

    recovery.add_checkpoint(param.name, window_id)


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-ensms", model=EnsmsConfig).load()
    print(cfg)
    recover = common.Recovery(cfg.root_dir, cfg, cfg.date, cfg.recover)
    last_checkpoint = recover.last_checkpoint()

    executor = (
        SynchronousExecutor()
        if cfg.n_par_compute == 1
        else QueueingExecutor(
            cfg.n_par_compute,
            cfg.window_queue_size,
            initializer=signal.signal,
            initargs=(signal.SIGTERM, signal.SIG_DFL),
        )
    )

    with executor:
        for param in cfg.parameters:
            out_key_kwargs = {"paramId": param.out_paramid} if param.out_paramid else {}
            window_manager = common.WindowManager(
                param.window_config(cfg.windows, cfg.steps),
                param.out_keys(cfg.out_keys, **out_key_kwargs),
            )

            if last_checkpoint:
                if param.name not in last_checkpoint:
                    print(f"Recovery: skipping completed param {param.name}")
                    continue
                checkpointed_windows = [
                    recover.checkpoint_identifiers(x)[1]
                    for x in recover.checkpoints
                    if param.name in x
                ]
                new_start = window_manager.delete_windows(checkpointed_windows)
                print(f"Recovery: param {param.name} looping from step {new_start}")
                last_checkpoint = None  # All remaining params have not been run

            requester = ParamRequester(
                param,
                cfg.sources,
                args.in_ens,
                cfg.members,
                cfg.total_fields,
            )
            iteration = functools.partial(ensms_iteration, cfg, param, recover)
            for keys, retrieved_data in parallel_data_retrieval(
                cfg.n_par_read,
                window_manager.dims,
                [requester],
                cfg.n_par_compute > 1,
                initializer=signal.signal,
                initargs=(signal.SIGTERM, signal.SIG_DFL),
            ):
                step = keys["step"]
                with ResourceMeter(f"Process step {step}"):
                    message_template, data = retrieved_data[0]

                    completed_windows = window_manager.update_windows(
                        keys,
                        data,
                    )
                    for window_id, accum in completed_windows:
                        executor.submit(iteration, window_id, accum, message_template)
            executor.wait()

    recover.clean_file()


if __name__ == "__main__":
    main(sys.argv)
