from __future__ import annotations
import contextlib
import contextvars
import functools
import json
from types import MethodType
from typing import Any, Dict, Optional

from sqlalchemy import Engine, MetaData, Table


__all__ = [
    "Model",
    "Property",
]

session_var = contextvars.ContextVar("sessions", default=None)

metadata = MetaData()


class LocalSession(object):
    def __init__(self, engine: Optional[Engine], metadata: MetaData) -> None:
        self._engine = engine
        self._conn = None
        self._metadata = metadata

    def create_all(self):
        self._metadata.create_all(self._engine)

    @contextlib.contextmanager
    def connect(self):
        if self._conn is None or self._conn.closed:
            # open a new connection if existing connection is closed
            self._conn = self._engine.connect()
            # close the connection at the end
            with self._conn:
                # begin a new transaction
                with self._conn.begin():
                    yield self._conn
        elif self._conn.in_transaction():
            # participates in the ongoing transaction
            with self._conn.begin_nested():
                yield self._conn
        else:
            # begin a new transaction
            with self._conn.begin():
                yield self._conn


class Model(object):
    __table__ = None

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.session: LocalSession = session_var.get()
        # Iterate through all attributes of the object
        for attr_name in dir(self):
            try:
                attr = getattr(self, attr_name)
            except AttributeError:
                continue
            # Check if the attribute is a callable (function or method)
            if not isinstance(attr, MethodType):
                continue
            setattr(self, attr_name, self._local_session_wrapper(attr))

    @property
    def table(self) -> Table:
        return type(self).__table__

    def _local_session_wrapper(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            token = session_var.set(self.session)
            try:
                # set the current local session to be observed by the
                # following function call
                return func(*args, **kwargs)
            finally:
                session_var.reset(token)

        return wrapper

    def to_dict(self) -> Dict[str, Any]:
        return {}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
