import logging
from conflator import CLIArg, ConfigModel, EnvVar
from pydantic import Field
from annotated_types import Annotated, Literal


class LoggingConfig(ConfigModel):
    level: Annotated[
        Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        CLIArg("--log"),
        EnvVar("LOG"),
        Field(description="Logging level [CRITICAL, ERROR, WARNING, INFO, DEBUG] (default: INFO)"),
    ] = "INFO"
    format: str = "%(asctime)s; %(name)s; %(levelname)s - %(message)s"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.basicConfig(format=self.format, level=self.level, force=True)
