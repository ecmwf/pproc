from typing import Optional

from pproc.configs.base import BaseConfig


class Config(BaseConfig):
    fc_date: Optional[str] = None
    type_em: Optional[str] = None
    type_es: Optional[str] = None
