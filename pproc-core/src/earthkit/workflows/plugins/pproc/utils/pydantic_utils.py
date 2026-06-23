from pydantic import BaseModel, ConfigDict


class PProcBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
