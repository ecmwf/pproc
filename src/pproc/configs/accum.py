from typing import Optional

from pproc.configs.base import BaseConfig


class Config(BaseConfig):
    class ParamConfig(BaseConfig.ParamConfig):
        vmin: Optional[float] = None
        vmax: Optional[float] = None

    date: Optional[str] = None
