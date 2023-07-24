from collections.abc import Iterator
import datetime
import functools
import itertools
import typing

import pandas as pd
from pyarrow import dataset as ds
import pyarrow as pa
import pydantic

from mxdesign.callbacks.base import CallbackList
from mxdesign.core.schema import Schema

try:
    # if datasets is installed, import it
    import datasets as hfd
except ImportError:
    hfd = None

__all__ = [
    'HFDatasetIterator',
    'BatchIterator',
    'create_batches',
]


BatchRootType = typing.TypeVar('BatchRootType')


# sentinel values
raise_error = object()


class Batch(
    Schema,
    pydantic.RootModel[typing.Iterable[BatchRootType]],
    typing.Generic[BatchRootType],
):
    root: typing.Iterable[BatchRootType]

    def __class_getitem__(cls, item):
        if not isinstance(item, tuple):
            item = item,
        cls.__args__ = item
        return cls


class HFDatasetIterator(object):
    def __init__(self, dataset) -> None:
        # : typing.Union[hfd.Dataset, hfd.DatasetDict]
        self._dataset = dataset
        self._length = None

    def __len__(self) -> int:
        if self._length is None:
            self._length = self._compute_length()
        return self._length

    def _compute_length(self) -> int:
        if isinstance(self._dataset, hfd.DatasetDict):
            return sum(ds.num_rows for ds in self._dataset.values())
        return self._dataset.num_rows

    @property
    def schema(self) -> pa.Schema:
        """Return the schema."""
        if isinstance(self._dataset, hfd.DatasetDict):
            input_schema = pa.unify_schemas(
                [ds._data.schema.with_metadata({})
                 for ds in self._dataset.values()]
            )
        else:
            input_schema = self._dataset._data.schema.with_metadata({})
        base_schema = pa.schema([('split', pa.string())])
        return pa.unify_schemas([base_schema, input_schema])

    def __iter__(self):
        if isinstance(self._dataset, hfd.DatasetDict):
            for key, hfds in self._dataset.items():
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                for item in hfds:
                    if 'split' not in item:
                        item['split'] = key
                    yield item
            return
        yield from self._dataset


class BatchIterator(Iterator[pa.RecordBatch]):
    def __init__(
        self, batches: Iterator[pa.RecordBatch],
        num_rows: typing.Optional[int] = None,
        schema: typing.Optional[pa.Schema] = None,
    ):
        super(BatchIterator, self).__init__()
        self._batches = iter(batches)
        # consumed number of rows
        self._consumed_rows = 0
        # total number of rows
        self._num_rows = num_rows
        # schema, None means to be determined
        self._schema = schema
        self.callbacks = CallbackList()

    def __next__(self) -> pa.RecordBatch:
        """Return the next batch."""
        self.callbacks.on_iter_begin(total=self.num_rows)
        try:
            batch = next(self._batches)
        except StopIteration as ex:
            self.callbacks.on_iter_end()
            raise ex
        n = batch.num_rows
        self._consumed_rows += n
        self.callbacks.on_iter_step(n)
        return batch

    def __iter__(self) -> typing.Generator[pa.RecordBatch, None, None]:
        """Iterate over the batches."""
        return self

    @property
    def schema(self) -> pa.Schema:
        """Return the schema."""
        if self._schema is None:
            self._batches, schema_iter = itertools.tee(self._batches)
            # this is really slow for large datasets
            # schema = pa.unify_schemas([batch.schema for batch in schema_iter])
            schema = None
            # determine schema from first batch
            for batch in schema_iter:
                schema = batch.schema
                break
            self._schema = schema
        return self._schema

    @property
    def num_rows(self) -> int:
        """Return the total number of rows."""
        if self._num_rows is None:
            raise TypeError(
                'TypeError: object of type '
                f'\'{type(self).__name__}\' has no num_rows'
            )
        return self._num_rows

    def _map_generator(self, func: typing.Callable, batched: bool = False) \
            -> typing.Generator[pa.RecordBatch, None, None]:
        for record_batch in self:
            if batched:
                table = pa.Table.from_batches([record_batch])
                batch_results = func(table)
            else:
                batch = record_batch.to_pylist()
                batch_results = []
                for item in batch:
                    batch_results.append(func(item))
            if isinstance(batch_results, pa.RecordBatch):
                yield batch_results
            if isinstance(batch_results, pa.Table):
                yield from batch_results.to_batches()
            elif isinstance(batch_results, pd.DataFrame):
                yield pa.RecordBatch.from_pandas(batch_results)
            elif isinstance(batch_results, dict):
                yield pa.RecordBatch.from_pydict(batch_results)
            else:
                yield pa.RecordBatch.from_pylist(batch_results)

    def map(self, func: typing.Callable, batched: bool = False) \
            -> 'BatchIterator':
        """Apply a function to each batch in the dataset."""
        schema, is_batched = inspect_mapping_func(func, default=None)
        batches = self._map_generator(func, batched=batched or is_batched)
        return BatchIterator(batches, num_rows=self.num_rows, schema=schema)

    def merge(self, other: 'BatchIterator') -> 'BatchIterator':
        """Merge two batch iterables."""
        batches = itertools.chain(self, other)
        schema = pa.unify_schemas([self.schema, other.schema])
        return BatchIterator(
            batches, num_rows=self.num_rows + other.num_rows, schema=schema
        )


def _batch_generator(iter, batch_size=1_000) \
        -> typing.Generator[pa.RecordBatch, None, None]:
    """Iterate over batches of an iterable."""
    if isinstance(iter, ds.Dataset):
        yield from iter.to_batches(batch_size=batch_size)
    elif isinstance(iter, pa.Table):
        yield from iter.to_batches(max_chunksize=batch_size)
    elif isinstance(iter, pa.RecordBatch):
        yield iter
    elif isinstance(iter, ds.Scanner):
        yield from iter.to_batches()
    else:
        batch = []
        for item in iter:
            batch.append(item)
            if len(batch) >= batch_size:
                yield pa.RecordBatch.from_pylist(batch)
                batch = []
        if len(batch) > 0:
            yield pa.RecordBatch.from_pylist(batch)


def create_batches(iter, batch_size=1_000) -> 'BatchIterator':
    # convert original iterator and extract number of records
    # list, scanner, dataset -> pyarrow record batches
    schema = None
    num_rows = None
    if isinstance(iter, HFDatasetIterator):
        num_rows = len(iter)
        schema = iter.schema
    elif isinstance(iter, ds.Dataset):
        num_rows = iter.count_rows()
        schema = iter.schema
    elif isinstance(iter, pa.Table):
        num_rows = iter.num_rows
        schema = iter.schema
    elif isinstance(iter, pa.RecordBatch):
        num_rows = iter.num_rows
        schema = iter.schema
    elif isinstance(iter, ds.Scanner):
        num_rows = iter.count_rows()
    elif hasattr(iter, '__len__'):
        num_rows = len(iter)
    batches = _batch_generator(iter, batch_size=batch_size)
    return BatchIterator(batches, num_rows=num_rows, schema=schema)


def inspect_mapping_func(func, default=raise_error):
    # check if func is partial function
    if isinstance(func, functools.partial):
        func = func.func
    type_hints = typing.get_type_hints(func)
    if 'return' in type_hints:
        output_type = type_hints['return']
        is_batched = False
        if issubclass(output_type, Batch):
            is_batched = True
            if hasattr(output_type, '__args__') and \
                    len(output_type.__args__) > 0:
                output_type = output_type.__args__[0]
        if issubclass(output_type, Schema):
            schema = create_arrow_schema(output_type)
            return schema, is_batched
    if default is not raise_error:
        return default, False
    raise TypeError(
        f'cannot determine output schema for function \'{func.__name__}\''
    )


def create_arrow_type(pytype: typing.Any, /) -> pa.DataType:
    """Return the arrow data type."""
    if isinstance(pytype, pa.DataType):
        return pytype
    if hasattr(pytype, '__origin__'):
        base_type = pytype.__origin__
        if base_type == typing.Union:
            # return pa.dense_union([
            #     pa.field(f'union_{t}', create_arrow_type(arg))
            #     for t, arg in enumerate(__type.__args__)
            #     if not isinstance(arg, type(None))
            # ])
            raise NotImplementedError
        if issubclass(base_type, typing.Sequence) \
                and not issubclass(base_type, str):
            dtype = create_arrow_type(pytype.__args__[0])
            return pa.list_(dtype)
        if issubclass(base_type, typing.Mapping):
            dtype = create_arrow_type(pytype.__args__[1])
            return pa.map_(pa.string(), dtype)
    elif isinstance(None, pytype):
        return pa.null()
    elif pytype == str:
        return pa.string()
    elif pytype == int:
        return pa.int64()
    elif pytype == float:
        return pa.float64()
    elif pytype == bool:
        return pa.bool_()
    elif pytype == bytes:
        return pa.binary()
    elif pytype == datetime.datetime:
        return pa.timestamp('ns')
    elif issubclass(pytype, Schema):
        return create_arrow_struct(pytype)
    return pa.from_numpy_dtype(pytype)


def create_arrow_field(name: str, pytype: typing.Any, /) -> pa.Field:
    if hasattr(pytype, '__origin__'):
        if len(pytype.__args__) == 2 and \
                isinstance(None, pytype.__args__[1]):
            # Optional[T] -> Union[T, None]
            dtype = create_arrow_type(pytype.__args__[0])
            return pa.field(name, dtype, nullable=True)
    return pa.field(name, create_arrow_type(pytype))


def create_arrow_struct(schema: typing.Type[Schema], /) -> pa.StructType:
    type_hints = typing.get_type_hints(schema)
    return pa.struct([
        create_arrow_field(name, type_hints[name])
        for name in schema.model_fields
    ])


def create_arrow_schema(schema: typing.Type[Schema], /) -> pa.Schema:
    type_hints = typing.get_type_hints(schema)
    return pa.schema([
        create_arrow_field(name, type_hints[name])
        for name in schema.model_fields
    ])
