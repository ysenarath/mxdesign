from typing import Any, Dict, Optional
from sqlalchemy import Column, Integer, String, Table, insert, select, update

from mxdesign.model.base import Model, metadata
from mxdesign.model.pagination import Pagination
from mxdesign.model.run import Run
from mxdesign.model.utils import NA

__all__ = [
    "Experiment",
]


class Experiment(Model):
    __table__ = Table(
        "experiment",
        metadata,
        # id is the id of the Expeiment
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("name", String(128), unique=True),
        Column("description", String(512)),
    )

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        id: Optional[int] = None,
    ):
        super(Experiment, self).__init__()
        if id is None:
            with self.session.connect() as conn:
                row = conn.execute(
                    self.table.select().where(self.table.c.name == name)
                ).first()
                if row is not None:
                    raise ValueError(
                        "found an existing Experiment with the name '{name}'"
                    )
                stmt = insert(self.table).values(
                    name=name,
                    description=description,
                )
                result = conn.execute(stmt)
                conn.commit()
                id = result.inserted_primary_key[0]
        self._id = id
        self._name = name
        self._description = description

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @name.setter
    def name(self, value: str):
        with self.session.connect() as conn:
            stmt = (
                update(self.table).where(self.table.c.id == self.id).values(name=value)
            )
            _ = conn.execute(stmt)
            conn.commit()
        self._name = value

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

    def start_run(
        self,
        name: Optional[str],
        description: Optional[str] = None,
    ):
        # name of the run
        return Run(
            name=name,
            description=description,
            experiment_id=self.id,
        )

    def list_runs(self, name: str = NA) -> Pagination[Run]:
        # populate existing object
        run_table = Run.__table__
        stmt = run_table.select()
        if name is not NA:
            stmt = stmt.where(run_table.c.name == name)
        stmt = stmt.where(run_table.c.experiment_id == self.id).order_by(
            run_table.c.created_at
        )
        return Pagination[Run](stmt)

    def get_run(self, id: int) -> Run:
        table = Run.__table__
        with self.session.connect() as conn:
            row = conn.execute(select(table).where(table.c.id == id)).first()
            if row is not None:
                return Run(**row._mapping)
            raise ValueError(f"Run with id {id} not found.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
        }
