"""Request and response models of the organizations domain."""

from pydantic import BaseModel, Field


class OrganizationRefusal(BaseModel):
    """The body of a request the organizations domain refuses."""

    detail: str = Field(description="Why the request was refused.")
