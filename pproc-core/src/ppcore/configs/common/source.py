from typing import Optional, Literal

from pydantic import BaseModel


class FDBSource(BaseModel):
    name: Literal["fdb"] = "fdb"
    config: Optional[str] = None
    stream: bool = True


class MARSSource(BaseModel):
    name: Literal["mars"] = "mars"


class FileSource(BaseModel):
    name: Literal["file"] = "file"
    path: str


class FilePatternSource(BaseModel):
    name: Literal["file-pattern"] = "file-pattern"
    pattern: str
    hive_partitioning: bool = True
