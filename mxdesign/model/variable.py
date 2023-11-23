from __future__ import annotations
from typing import Any, Dict, Optional, Union
from sqlalchemy import (
    Column,
    Index,
    Integer,
    String,
    ForeignKey,
    Table,
    insert,
    text,
)
from sqlalchemy.exc import IntegrityError

from mxdesign.model.base import Model, metadata
from mxdesign.model.utils import get_type

__all__ = [
    "Variable",
]


class Variable(Model):
    __table__ = Table(
        "variable",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("name", String, nullable=False),
        Column("value", String, nullable=False),
        Column("type", String, nullable=False),
        Column("dtype", String, nullable=False),
        Column("step", Integer, nullable=True),
        Column("run_id", ForeignKey("run.id"), nullable=False),
        Index(
            "idx_name_step_run_id",
            "name",
            text("COALESCE(step,'-1')"),
            "run_id",
            unique=True,
        ),
    )

    def __init__(
        self,
        name: str,
        value: Union[str, int, float],
        type: str,
        step: int,
        run_id: int,
        id: Optional[int] = None,
        dtype: Optional[str] = None,
    ):
        super(Variable, self).__init__()
        if dtype is None:
            dtype = str(get_type(value).__name__)
        if not dtype in {"int", "float", "str"}:
            raise TypeError(f"value must be a int, float or string, found {dtype}")
        if id is None:
            with self.session.connect() as conn:
                stmt = insert(self.__table__).values(
                    name=name,
                    value=str(value),
                    type=type,
                    step=step,
                    dtype=dtype,
                    run_id=run_id,
                )
                error_message = None
                try:
                    result = conn.execute(stmt)
                except IntegrityError as ex:
                    error_message: str = ex.args[0]
                if (
                    error_message is not None
                    and "idx_name_step_run_id" in error_message
                ):
                    raise ValueError(f"variable with name '{name}' exists")
                id = result.inserted_primary_key[0]
                conn.commit()
        self._id = id
        self._name = name
        self._value = value
        self._dtype = dtype
        self._type = type
        self._step = step
        self._run_id = run_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        return self._type

    @property
    def value(self) -> Union[int, float, str]:
        if self._dtype == "int":
            return int(self._value)
        if self._dtype == "float":
            return float(self._value)
        return str(self._value)

    @property
    def step(self) -> int:
        return self._step

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "name": self._name,
            "value": self.value,
            "type": self._type,
            "step": self._step,
            "run_id": self._run_id,
        }
