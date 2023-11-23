from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    create_engine,
    Table,
    Column,
    Integer,
    String,
    MetaData,
    insert,
)
from mxdesign.model.environment import Environment
from mxdesign.model.pagination import Pagination

from mxdesign.model.base import LocalSession, Model


metadata = MetaData()


class EnvironmentLoader(Model):
    __table__ = Table(
        "environment",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("url", String, nullable=False),
    )

    def __init__(
        self,
        url: str,
        id: Optional[int] = None,
    ):
        super(EnvironmentLoader, self).__init__()
        if id is None:
            with self.session.connect() as conn:
                row = conn.execute(
                    self.table.select().where(self.table.c.url == url)
                ).first()
                if row is not None:
                    raise ValueError(f"found an existing User environment")
                stmt = insert(self.table).values(
                    url=url,
                )
                result = conn.execute(stmt)
                conn.commit()
                id = result.inserted_primary_key[0]
        self._id = id
        self._url = url

    @property
    def id(self) -> int:
        return self._id

    @property
    def url(self) -> str:
        return self._url

    def load(self) -> Environment:
        return Environment(self.url)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "url": self._url,
        }


class User(Model):
    __table__ = Table(
        "user",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("username", String(50), unique=True, nullable=False),
        Column("password", String(512), nullable=False),
        Column("email", String(100), unique=True, nullable=True),
        Column("role", String(50), nullable=False, default="default"),
        Column("diabled", Boolean, nullable=False, default=False),
    )

    def __init__(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        role: Optional[str] = "default",
        diabled: bool = False,
        id: Optional[int] = None,
    ):
        super(User, self).__init__()
        if id is None:
            with self.session.connect() as conn:
                row = conn.execute(
                    self.table.select().where(self.table.c.username == username)
                ).first()
                if row is not None:
                    raise ValueError(
                        f"found an existing User with the username '{username}'"
                    )
                stmt = insert(self.table).values(
                    username=username,
                    password=password,
                    email=email,
                    role=role,
                    diabled=diabled,
                )
                result = conn.execute(stmt)
                conn.commit()
                id = result.inserted_primary_key[0]
        self._username = username
        self._password = password
        self._email = email
        self._diabled = diabled
        self._role = role
        self._id = id

    @property
    def id(self) -> int:
        return self._id

    @property
    def username(self) -> str:
        return self._username

    @property
    def password(self) -> str:
        return self._password

    @property
    def hashed_password(self) -> str:
        return self._password

    @property
    def email(self) -> Optional[str]:
        return self._email

    @property
    def diabled(self) -> bool:
        return self._diabled

    @property
    def role(self) -> bool:
        return self._role

    def to_dict(self) -> Dict[str, Any]:
        return {
            "username": self._username,
            "password": self._password,
            "email": self._email,
            "role": self._role,
            "diabled": self._diabled,
            "id": self._id,
        }


class Database(Model):
    def __init__(self, url: str) -> None:
        super(Database, self).__init__()
        self.session = LocalSession(
            engine=create_engine(url),
            metadata=metadata,
        )
        self.session.create_all()

    def create_user(self, username: str, password: str, role: str = "default") -> User:
        return User(
            username=username,
            password=password,
            role=role,
        )

    def get_user(self, username: str) -> User:
        user_table = User.__table__
        user = None
        with self.session.connect() as conn:
            row = conn.execute(
                user_table.select().where(user_table.c.username == username)
            ).first()
            if row is not None:
                user = User(**row._mapping)
        return user

    def add_environment(self, url: str) -> EnvironmentLoader:
        return EnvironmentLoader(url=url)

    def get_environment(self, id: int) -> Environment:
        env_loader_table = EnvironmentLoader.__table__
        loader = None
        with self.session.connect() as conn:
            row = conn.execute(
                env_loader_table.select().where(env_loader_table.c.id == id)
            ).first()
            if row is not None:
                loader = EnvironmentLoader(**row._mapping)
        if loader is None:
            return None
        return loader.load()

    def list_environments(self) -> Pagination[EnvironmentLoader]:
        stmt = EnvironmentLoader.__table__.select()
        return Pagination[EnvironmentLoader](stmt)
