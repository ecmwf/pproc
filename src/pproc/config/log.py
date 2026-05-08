# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from annotated_types import Annotated, Literal
from conflator import CLIArg, ConfigModel, EnvVar
from pydantic import Field


class LoggingConfig(ConfigModel):
    level: Annotated[
        Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        CLIArg("--log"),
        EnvVar("LOG"),
        Field(
            description="Logging level [CRITICAL, ERROR, WARNING, INFO, DEBUG] (default: INFO)"
        ),
    ] = "INFO"
    format: str = "%(asctime)s; %(name)s; %(levelname)s - %(message)s"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.basicConfig(format=self.format, level=self.level, force=True)
