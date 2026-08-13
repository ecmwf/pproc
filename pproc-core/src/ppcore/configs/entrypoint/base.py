# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import yaml
from annotated_types import Annotated
from pydantic import Field, BeforeValidator
from conflator import ConfigModel, CLIArg

from ppcore.configs.entrypoint.log import LoggingConfig
from ppcore.configs.entrypoint.execution import ExecutionConfig
from ppcore.configs.product import ProductConfig
from ppcore.utils.entrypoint import validate_overrides


class EntrypointConfig(ConfigModel):
    log: LoggingConfig = LoggingConfig()
    execution: ExecutionConfig = ExecutionConfig()
    products: list[ProductConfig]
    input_overrides: Annotated[
        dict,
        BeforeValidator(validate_overrides),
        CLIArg(
            "--override-input",
            action="append",
            default=[],
            metavar="KEY=VALUE,...",
        ),
        Field(
            default_factory=dict, description="Override input requests with these keys"
        ),
    ]
    output_overrides: Annotated[
        dict,
        BeforeValidator(validate_overrides),
        CLIArg(
            "--override-output",
            action="append",
            default=[],
            metavar="KEY=VALUE,...",
        ),
        Field(default_factory=dict, description="Override outputs with these keys"),
    ]

    def dump(self) -> str:
        return yaml.dump(self.model_dump(by_alias=True), sort_keys=False)
