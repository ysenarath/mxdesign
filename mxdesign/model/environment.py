from typing import Any, Dict, Optional

from sqlalchemy import create_engine, select
from mxdesign.model.base import Model, LocalSession, metadata

from mxdesign.model.experiment import Experiment
from mxdesign.model.pagination import Pagination

__all__ = [
    "Environment",
]


class Environment(Model):
    def __init__(self, url: str) -> None:
        super(Environment, self).__init__()
        self.session = LocalSession(
            engine=create_engine(url),
            metadata=metadata,
        )
        self.session.create_all()

    def get_experiment(self, id: int):
        expr_table = Experiment.__table__
        with self.session.connect() as conn:
            row = conn.execute(expr_table.select().where(expr_table.c.id == id)).first()
            if row is None:
                raise ValueError("Experiment with id {id} does not exist")
            experiment = Experiment(**row._mapping)
        return experiment

    def get_experiment_by_name(self, name: str):
        expr_table = Experiment.__table__
        with self.session.connect() as conn:
            row = conn.execute(
                expr_table.select().where(expr_table.c.name == name)
            ).first()
            if row is None:
                raise ValueError("Experiment with name '{name}' does not exist")
            experiment = Experiment(**row._mapping)
        return experiment

    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
    ):
        return Experiment(
            name=name,
            description=description,
        )

    def get_or_create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
    ):
        expr_table = Experiment.__table__
        with self.session.connect() as conn:
            row = conn.execute(
                expr_table.select().where(expr_table.c.name == name)
            ).first()
            if row is None:
                experiment = self.create_experiment(
                    name=name,
                    description=description,
                )
            else:
                experiment = Experiment(**row._mapping)
        return experiment

    def list_experiments(self) -> Pagination[Experiment]:
        query = select(Experiment.__table__)
        return Pagination[Experiment](query=query)

    def to_dict(self) -> Dict[str, Any]:
        return {}
