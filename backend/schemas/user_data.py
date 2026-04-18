from pydantic import BaseModel, Field


class MyDataUpdateRequest(BaseModel):
    data: dict = Field(default_factory=dict)
