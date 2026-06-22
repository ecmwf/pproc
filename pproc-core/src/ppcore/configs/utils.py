from pydantic import BaseModel, ConfigDict


class PProcCoreBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
