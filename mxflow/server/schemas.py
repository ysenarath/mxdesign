from __future__ import annotations
import datetime
from typing import List, Optional, Union

from pydantic import BaseModel

from mxdesign.model.base import Model


class Schema(BaseModel):
    @classmethod
    def from_orm(cls, obj: Model) -> Schema:
        return cls(**obj.to_dict())


class ExperimentSchema(Schema):
    id: int
    name: str
    description: Optional[str]


class RunSchema(Schema):
    id: int
    name: str
    experiment_id: int
    description: Optional[str]
    created_at: Optional[datetime.datetime]
    updated_at: Optional[datetime.datetime]
    parent_run_id: Optional[int]


class VariableSchema(Schema):
    id: int
    name: str
    value: Union[str, int, float]
    type: str
    step: Optional[int]
    run_id: int


class UserSchema(Schema):
    username: str
    email: Union[str, None] = None
    disabled: Union[bool, None] = None
    id: Optional[int] = None


class TokenSchema(Schema):
    access_token: str
    token_type: str


class TokenDataSchema(Schema):
    username: Union[str, None] = None
    scopes: List[str] = []


class UserInDBSchema(UserSchema):
    password: str
    role: str = "default"


class EnvironmentLoaderSchema(Schema):
    url: str


class EnvironmentLoaderInDBSchema(EnvironmentLoaderSchema):
    id: int
