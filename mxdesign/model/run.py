from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Index,
    Integer,
    String,
    ForeignKey,
    Table,
    insert,
    select,
    text,
    update,
)

from mxdesign.model.base import Model, metadata
from mxdesign.model.pagination import Pagination
from mxdesign.model.utils import NA
from mxdesign.model.variable import Variable

__all__ = [
    "Run",
]


class Run(Model):
    __table__ = Table(
        "run",
        metadata,
        # id is the id of the Expeiment
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("name", String(128), nullable=False),
        Column("description", String(512), nullable=True),
        Column("created_at", DateTime, default=datetime.utcnow),
        Column(
            "updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
        ),
        Column(
            "experiment_id",
            ForeignKey("experiment.id"),
            nullable=False,
        ),
        Column(
            "parent_run_id",
            ForeignKey("run.id"),
            nullable=True,
        ),
        Index(
            "idx_name_experiment_id_parent_run_id",
            "name",
            "experiment_id",
            text("COALESCE(parent_run_id,'-1')"),
            unique=True,
        ),
    )

    def __init__(
        self,
        name: str,
        experiment_id: int,
        description: Optional[str] = None,
        id: Optional[int] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        parent_run_id: Optional[int] = None,
    ):
        super(Run, self).__init__()
        if id is None:
            with self.session.connect() as conn:
                result = conn.execute(
                    self.table.select().where(self.table.c.name == name)
                ).first()
                if result is not None:
                    raise ValueError(
                        f"Run with name '{name}' exists for Experiment with id '{experiment_id}'"
                    )
                stmt = insert(self.table).values(
                    name=name,
                    description=description,
                    experiment_id=experiment_id,
                    parent_run_id=parent_run_id,
                )
                result = conn.execute(stmt)
                id = result.inserted_primary_key[0]
                conn.commit()
        self._id = id
        self._name = name
        self._description = description
        self._experiment_id = experiment_id
        self._created_at = created_at
        self._updated_at = updated_at
        self._parent_run_id = parent_run_id

    @property
    def id(self) -> int:
        return self._id

    @property
    def experiment_id(self) -> int:
        return self._experiment_id

    @property
    def parent_run_id(self) -> int:
        return self._parent_run_id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        with self.session.connect() as conn:
            stmt = (
                update(self.table).where(self.table.c.id == self.id).values(name=value)
            )
            _ = conn.execute(stmt)
            conn.commit()
        self._name = value

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, value: str):
        with self.session.connect() as conn:
            stmt = (
                update(self.table)
                .where(self.table.c.id == self.id)
                .values(description=value)
            )
            _ = conn.execute(stmt)
            conn.commit()
        self._description = value

    @property
    def updated_at(self) -> datetime:
        return self._updated_at

    @property
    def created_at(self) -> datetime:
        return self._created_at

    def log_metric(
        self,
        name: str,
        value: str,
        step: Optional[int] = None,
    ):
        return Variable(
            name=name,
            value=value,
            type="metric",
            step=step,
            run_id=self.id,
        )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    def log_param(
        self,
        name: str,
        value: str,
        step: Optional[int] = None,
    ):
        return Variable(
            name=name,
            value=value,
            type="param",
            step=step,
            run_id=self.id,
        )

    def log_params(self, metrics: Dict[str, Any], step: Optional[int] = None):
        for key, value in metrics.items():
            self.log_param(key, value, step=step)

    def get_variable(self, id: int) -> Variable:
        table = Variable.__table__
        with self.session.connect() as conn:
            row = conn.execute(select(table).where(table.c.id == id)).first()
            if row is not None:
                return Variable(**row._mapping)
            else:
                raise ValueError(f"Variable with id {id} not found")

    def list_variables(self) -> List[Variable]:
        var_table = Variable.__table__
        vars = []
        with self.session.connect() as conn:
            rows = conn.execute(select(var_table))
            for row in rows:
                var = Variable(**row._mapping)
                vars.append(var)
        return vars

    def start_run(
        self,
        name: Optional[str],
        description: Optional[str] = None,
    ):
        # name of the run
        return Run(
            name=name,
            description=description,
            experiment_id=self.experiment_id,
            parent_run_id=self.id,
        )

    def list_runs(self, name: str = NA) -> Pagination[Run]:
        # populate existing object
        run_table = Run.__table__
        stmt = run_table.select()
        if name is not NA:
            stmt = stmt.where(run_table.c.name == name)
        stmt = (
            stmt.where(run_table.c.experiment_id == self.experiment_id)
            .where(run_table.c.parent_run_id == self.id)
            .order_by(run_table.c.created_at)
        )
        return Pagination[Run](stmt)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "experiment_id": self._experiment_id,
            "created_at": self._created_at,
            "updated_at": self._updated_at,
            "parent_run_id": self._parent_run_id,
        }
