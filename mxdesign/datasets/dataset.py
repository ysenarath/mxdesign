from collections.abc import Mapping, Sequence, Iterable
from dataclasses import field
import typing
from pathlib import Path
import uuid

import numpy as np
import numcodecs
import pandas as pd
import zarr

from mxdesign.core.schema import Schema
from mxdesign.core.version import Version
from mxdesign.utils.cache import getcachedir

__all__ = [
    'Dataset',
    'DatasetInfo',
]

DEFAULT_BATCH_SIZE = 1000

StoreOrPath = typing.Union[zarr.storage.BaseStore, str, Path, None]
RBVT = typing.Union[zarr.Array, np.ndarray, 'FieldMapping', 'RecordBatch']


class DatasetInfo(Schema):
    name: typing.Optional[str] = None
    description: typing.Optional[str] = None
    version: Version = Version('0.0.1')
    citation: typing.Optional[str] = None
    license: typing.Optional[str] = None
    splits: typing.List[str] = field(default_factory=list)
    aliases: typing.List[str] = field(default_factory=list)


def prepare(data, path='$', record=False) -> typing.Tuple[int, typing.Any]:
    '''Prepare data for writing to zarr.'''
    if data is None:
        return 0, None
    elif isinstance(data, pd.Series) and data.name is None:
        # => array-like
        return len(data), data.array
    elif isinstance(data, pd.Series):
        # => {column -> array-like}
        return len(data), {
            # no copy array dict
            data.name: data.array
        }
    elif isinstance(data, pd.DataFrame):
        # => {column -> array-like}
        return len(data), {
            # no copy array dict
            col: data[col].array for col in data.columns
        }
    elif isinstance(data, np.ndarray) and len(data.shape) == 0:
        return 0, data.item()
    elif isinstance(data, np.ndarray):
        # dont do anything for numpy types
        # => array-like
        return data.shape[0], data
    elif isinstance(data, Mapping):
        dict_of_lists = {}
        unified_length = None
        for key, value in data.items():
            value_path = path + '.' + key.replace('.', '\\.')
            length, value = prepare(value, path=value_path)
            if unified_length is None:
                unified_length = length
            elif unified_length != length and not record:
                raise ValueError(
                    f'expected {unified_length} rows, '
                    f'got {length} for {value_path}'
                )
            if isinstance(value, Mapping):
                for k, v in value.items():
                    dict_of_lists[(key,) + k] = v
            else:
                dict_of_lists[(key,)] = value
        return unified_length, dict_of_lists
    elif isinstance(data, Iterable) and not isinstance(data, str):
        list_of_dict = []
        keys = set()
        is_all_mapping = True
        for idx, item in enumerate(data):
            _, item = prepare(item, path=f'{path}[{idx}]', record=True)
            if is_all_mapping and isinstance(item, Mapping):
                keys = keys.union(item.keys())
            elif item is not None:
                is_all_mapping = False
            list_of_dict.append(item)
        length = len(list_of_dict)
        if is_all_mapping:
            dict_of_lists = {}
            for doc in list_of_dict:
                for key in keys:
                    dict_of_lists \
                        .setdefault(key, []) \
                        .append(doc.get(key))
            return length, dict_of_lists
        return length, list_of_dict
    # numpy scalers
    return 0, data


class ValueInfo(typing.NamedTuple):
    shape: typing.Tuple[int, ...]
    dtype: typing.Any
    codec: typing.Any


def value_info(value):
    if isinstance(value, np.ndarray):
        value_shape = value.shape
        value_dtype = value.dtype
    else:
        value_shape = (0,)  # 1-d array
        value_dtype = object  # object dtype
        # d-type from the first non-None value
        for item in value:
            if item is None:
                continue
            if isinstance(item, np.ndarray):
                value_shape = (0,) + item.shape
                value_dtype = item.dtype
            if np.isscalar(item):
                # if value is not a scaler
                # keep the object dtype
                value_dtype = type(item)
            break
    codec = None
    if value_dtype == object or isinstance(value_dtype, np.dtypes.ObjectDType):
        codec = numcodecs.MsgPack()
    return ValueInfo(value_shape, value_dtype, codec)


def find_array_paths(group: zarr.Group, paths=[]) -> typing.Set[str]:
    def visit(path, obj):
        if not isinstance(obj, zarr.Array):
            return
        paths.append(path)
    group.visititems(visit)
    return set(paths)


class RecordBatch(Mapping):
    def __init__(self, data=None, parent=None):
        self._data: typing.Dict[str, RBVT] = data or {}
        self._parent: typing.Optional[RecordBatch] = parent

    @property
    def _root(self) -> 'RecordBatch':
        if self._parent is None:
            return self
        return self._parent._root

    @property
    def num_records(self):
        if self._parent is not None:
            return self._root.num_records
        for value in self.values():
            if isinstance(value, RecordBatch):
                return value.num_records
            return value.shape[0]
        return 0

    def __getitem__(self, key) -> RBVT:
        if isinstance(key, tuple):
            value = self._data
            for k in key:
                value = value[k]
        else:
            value = self._data[key]
        if isinstance(value, Mapping):
            return RecordBatch(value, parent=self)
        return value

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            key_root, key_suffix = key[0], key[1:]
            if key_root not in self._data:
                self._data[key_root] = {}
            self[key_root][key_suffix] = value
        elif isinstance(value, Mapping):
            if key not in self._data:
                self._data[key] = {}
            for k, v in value.items():
                self[key][k] = v
        else:
            if value.shape[0] != self.num_records:
                raise ValueError(
                    f'expected {self.num_records} rows, '
                    f'got {value.shape[0]}'
                )
            self._data[key] = value

    def __iter__(self):
        return iter(self._data.keys())

    def __len__(self) -> int:
        return len(self._data)

    def to_list(self):
        # convert this struct of lists to a list of structs
        dict_of_lists = {}
        num_records = None
        for key, value in self.items():
            if isinstance(value, RecordBatch):
                value = value.to_list()
            if num_records is None:
                num_records = len(value)
            dict_of_lists[key] = value
        if num_records is None:
            return []
        result = []
        for i in range(num_records):
            record = {}
            for key, value in dict_of_lists.items():
                record[key] = value[i]
            result.append(record)
        return result

    # alias
    tolist = to_list


class ILoc(object):
    def __init__(self, data: 'FieldMapping'):
        self._data: FieldMapping = data

    def __getitem__(self, item):
        if isinstance(item, int):
            batch = False
        elif isinstance(item, slice) or (
            isinstance(item, Sequence) and not isinstance(item, str)
        ):
            batch = True
        else:
            raise TypeError(
                'cannot slice a \'FieldMapping\' using '
                f'a \'{type(item).__name__}\''
            )
        value = {}
        for key in self._data.keys():
            field = self._data[key]
            if isinstance(field, FieldMapping):
                record = field.iloc[item]
            else:
                record = field[item]
            value[key] = record
        if batch:
            return RecordBatch(value)
        return value

    def __setitem__(self, item, value):
        raise NotImplementedError

    def __repr__(self):
        return f'<ILoc {self._data}>'

    def __iter__(self):
        return iter(self._data.keys())


class FieldMapping(Mapping):
    def __init__(self, data, parent=None):
        self._data: zarr.Group = data
        self._parent: typing.Optional[FieldMapping] = parent

    @property
    def _root(self) -> 'FieldMapping':
        if self._parent is None:
            return self
        return self._parent._root

    @property
    def batch_size(self):
        if self._parent is not None:
            return self._root.batch_size
        for arr_key in self._data.array_keys():
            arr = self._data[arr_key]
            return arr.chunks[0]
        # return default chunk-size
        return DEFAULT_BATCH_SIZE

    @property
    def num_rows(self):
        # get the number of rows from the first array of root
        if self._parent is not None:
            # is not the root
            return self._root.num_rows
        # is the root -> get the size of the first array
        for arr_key in self._data.array_keys():
            arr = self._data[arr_key]
            return arr.shape[0]
        # no array found
        return 0

    def __getitem__(self, key) -> typing.Any:
        result = self._data[key]
        if isinstance(result, zarr.Group):
            return FieldMapping(result, parent=self)
        return result

    def __setitem__(self, key, value):
        """Add a field in the mapping."""
        length, data = prepare({key: value})
        if length != self.num_rows:
            raise ValueError(
                f'expected {self.num_rows} rows, got {length}'
            )
        data = {'/'.join(k): v for k, v in data.items()}
        for key, value in data.items():
            info = value_info(value)
            shape = (self.num_rows,) + info.shape[1:]
            chunks = (self.batch_size,) + info.shape[1:]
            # add fields under this group
            self._data.create_dataset(
                key, shape=shape,
                chunks=chunks,
                dtype=info.dtype,
                object_codec=info.codec,
            )
            self._data[key][:] = value

    def __iter__(self):
        return iter(self._data.keys())

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self):
        return f'<Fields {self._data}>'

    @property
    def iloc(self):
        return ILoc(self)

    def to_batches(self) -> typing.Iterator[RecordBatch]:
        batch_size = self.batch_size
        for idx in range(0, self.num_rows, batch_size):
            yield self.iloc[idx:idx + batch_size]

    def create_field(self, name) -> 'FieldMapping':
        self._data.create_group(name)
        return self[name]


class Dataset(object):
    def __init__(
        self, store: StoreOrPath = None,
        overwrite: str = False,
    ):
        super(Dataset, self).__init__()
        if store is None:
            store = zarr.storage.TempStore(
                prefix='dataset-', dir=getcachedir()
            )
        if not isinstance(store, zarr.storage.BaseStore):
            store = zarr.DirectoryStore(store)
        group: zarr.Group = zarr.group(
            store=store,
            overwrite=overwrite,
        )
        fields = group.create_group('fields')
        self.fields = FieldMapping(fields)
        self.cache = group.create_group('cache')

    def __len__(self):
        return self.fields.num_rows

    def __getitem__(self, item):
        return self.fields.iloc[item]

    def extend(self, data):
        # we assume that the preparing process will always return a dict
        # otherwise the input is invalid :)
        if not hasattr(data, '__len__') or len(data) > 1000:
            i = 0
            while True:
                try:
                    chunk = data[i:i + 1000]
                except TypeError:
                    chunk = []
                    for j in range(i, i + 1000):
                        try:
                            chunk.append(next(data))
                        except StopIteration:
                            break
                if len(chunk) == 0:
                    break
                self.extend(chunk)
                i += 1000
            return
        initial_length = len(self)
        length, data = prepare(data)
        # {str -> np.ndarray}
        data = {'/'.join(key): value for key, value in data.items()}
        array_keys = find_array_paths(self.fields._data) \
            .union(data.keys())
        for key in array_keys:
            # get ket from data it may or may not exist
            value = data.get(key, None)
            if key not in self.fields:
                # if the value is not in fields then it must be in data
                info = value_info(value)
                shape = (initial_length + length,) + info.shape[1:]
                chunks = (self.fields.batch_size,) + shape[1:]
                self.fields._data.create_dataset(
                    key, shape=shape,
                    chunks=chunks,
                    fill_value=None,
                    dtype=info.dtype,
                    object_codec=info.codec,
                )
                arr = self.fields[key]
            else:
                # expand the arrays to fit the new data
                arr = self.fields[key]
                shape = (initial_length + length,) + arr.shape[1:]
                arr.resize(*shape)
            # fill the arrays with new data
            if value is None:
                arr[initial_length:] = np.nan
                continue
            if len(arr.shape) == 1:
                # force the data to array of 1d objects
                value = np.array(value, dtype=arr.dtype)
            arr[initial_length:] = value

    def append(self, item):
        # extend by one item
        self.extend([item])

    def _extend_group(self, group: zarr.Group, batch: RecordBatch):
        for key, value in batch.items():
            if isinstance(value, RecordBatch):
                if key not in group:
                    group.create_group(key)
                self.extend(group[key], value)
            elif key in group:
                group[key].append(value)
            else:
                # assume value is array like
                info = value_info(value)
                chunks = (self.fields.batch_size,) + info.shape[1:]
                group.array(key, value, chunks=chunks)

    def map(self, func, batched=False) -> FieldMapping:
        if not batched:
            raise NotImplementedError
        cache_uuid = uuid.uuid4()
        group = self.cache.create_group(cache_uuid)
        for batch in self.fields.to_batches():
            batch_result = func(batch)
            self._extend_group(group, batch_result)
        return FieldMapping(group)

    def move(self, mapping: typing.Dict[str, str]):
        for source, dest in mapping.items():
            self.fields._data.move(source, dest)
