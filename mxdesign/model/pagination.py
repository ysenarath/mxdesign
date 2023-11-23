from __future__ import annotations
from collections.abc import Sequence
from typing import Any, Dict, Generator, Generic, List, TypeVar, get_origin

from sqlalchemy import func, select
from sqlalchemy.orm import Query

from mxdesign.model.base import Model

T = TypeVar("T", bound=Model)


class Pagination(Model, Sequence, Generic[T]):
    def __class_getitem__(cls, item):
        cls.__args__ = (item,)
        return cls

    # this is a lazy squence object
    def __init__(
        self,
        query: Query,
        page_number: int = 1,
        page_size: int = 10,
    ):
        super(Pagination, self).__init__()
        self._query = query
        self._page_number = page_number
        self._page_size = page_size
        self._cache = None

    @property
    def query(self) -> Query:
        return self._query

    @property
    def page_number(self) -> int:
        return self._page_number

    @page_number.setter
    def page_number(self, value: int):
        self._page_number = value
        self._cache = None

    @property
    def page_size(self) -> int:
        return self._page_size

    @page_size.setter
    def page_size(self, value: int):
        self._page_size = value
        self._cache = None

    def _get_cached_items(self):
        if self._cache is None:
            offset = (self.page_number - 1) * self.page_size
            query = self.query.limit(self.page_size).offset(offset)
            C = type(self).__args__[0]
            with self.session.connect() as conn:
                result = conn.execute(query)
                rows = result.fetchall()
                self._cache = [C(**row._mapping) for row in rows]
        return self._cache

    @property
    def items(self) -> List[T]:
        return self._get_cached_items()

    def __iter__(self) -> Generator[T, None, None]:
        yield from self.items

    def __getitem__(self, index) -> T:
        return self.items[index]

    def __len__(self) -> int:
        # number of itmes in this page
        return len(self.items)

    def get_total_items(self) -> int:
        count_query = select([func.count().label("count")]).select_from(
            self.query.alias()
        )
        with self.session.connect() as conn:
            result = conn.execute(count_query)
            total_items = result.scalar()
        return total_items

    def get_total_pages(self) -> int:
        total_items = self.get_total_items()
        return (total_items + self.page_size - 1) // self.page_size

    def next_page(self) -> Pagination[T]:
        C = type(self).__args__[0]
        return Pagination[C](
            self.query,
            self.page_number + 1,
            self.page_size,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [t.to_dict() for t in self],
            "page_number": self.page_number,
            "page_size": self.page_size,
        }
