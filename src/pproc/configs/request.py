import json
import os
from typing import List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Request(BaseModel):
    model_config = ConfigDict(
        extra="ignore", populate_by_name=True, coerce_numbers_to_str=True
    )

    class_: str = Field(alias="class")
    stream: str
    expver: str
    levtype: str
    levelist: Optional[Union[int, list]] = None
    domain: str
    param: str
    date: Union[str, List[str]]
    time: Union[str, List[str]]
    step: Union[str, int, List[str], List[int]]
    type_: str = Field(alias="type")
    number: Optional[Union[int, List[int]]] = None
    hdate: Optional[Union[str, List[str]]] = None
    quantile: Optional[Union[str, List[str]]] = None


def write_requests(output_file: str, output_reqs: List[Request]):
    req_dicts = [req.dict(by_alias=True, exclude_none=True) for req in output_reqs]
    _, extension = os.path.splitext(output_file)
    with open(output_file, "w") as f:
        if extension == ".json":
            json.dump(req_dicts, f, sort_keys=False, indent=2)
        else:
            yaml.dump(req_dicts, f, sort_keys=False)
