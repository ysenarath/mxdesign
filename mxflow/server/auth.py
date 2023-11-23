from datetime import timedelta, datetime
from typing import Union

from fastapi.security import (
    OAuth2PasswordBearer,
)
from jose import jwt
from passlib.context import CryptContext

from mxflow.config import config
from mxflow.server.schemas import (
    UserInDBSchema,
)
from mxflow.server.db import db

SECRET_KEY = config["server"]["secret_key"]
ALGORITHM = config["server"]["algorithm"]
ACCESS_TOKEN_EXPIRE_MINUTES = config["server"]["access_token_expire_minutes"]

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        # "me": "Read information about the current user.",
        # "items": "Read items.",
    },
)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(username: str) -> UserInDBSchema:
    user = db.get_user(username)
    return UserInDBSchema.from_orm(user)


def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    return encoded_jwt
