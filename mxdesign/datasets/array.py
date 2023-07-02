import typing
from collections.abc import Sequence, Mapping
import numpy as np

import zarr

__all__ = [
    'RecordBatch',
]


RT = typing.Union[zarr.Group, zarr.Array]


class ArrayDict(Mapping):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._data: typing.Dict[str, zarr.Array] = {}
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def _set_key_value(self, key: str, value: zarr.Array) -> None:
        if not isinstance(value, zarr.Array):
            value = zarr.array(value)
        self._data[key] = value

    def __getitem__(self, key: str) -> zarr.Array:
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self._data)


class DataArray(object):
    def __init__(self, data, coords=None, dims=None, name=None, chunks=None):
        #  a zarr.Array holding the array’s values
        self._data = zarr.creation.array(data, chunks=chunks)
        # dimension names for each axis (e.g., ('x', 'y', 'z'))
        if dims is None:
            dims = tuple(f'dim_{i}' for i in range(self._data.ndim))
        self._dims = dims
        # a dict-like container of arrays (coordinates) that label each point
        # (e.g., 1-dimensional arrays of numbers, datetime objects or strings)
        if coords is None:
            coords = {}
        if isinstance(coords, Sequence):
            coords = dict(zip(self._dims, coords))
        for idx, dim in enumerate(self._dims):
            if dim not in coords:
                continue
            coords_size = len(coords[dim])
            data_dim_shape = self._data.shape[idx]
            if dim in coords and coords_size != data_dim_shape:
                raise ValueError(
                    f'length of coordinates of dimention '
                    f'\'{dim}\' ({coords_size}) does not match '
                    f'length of data \'{data_dim_shape}\''
                )
        self._coords: Mapping[str, zarr.Array] = ArrayDict(coords)
        # name of the data variable.
        self._name = name
        # dict to hold arbitrary metadata (attributes)
        self._attrs = {}

    @property
    def coords(self) -> Mapping[str, zarr.Array]:
        """Return the coordinates."""
        return self._coords

    @property
    def dims(self) -> typing.Tuple[str, ...]:
        """Return the dimension names."""
        return self._dims

    @property
    def name(self):
        """Return the name of the data variable."""
        return self._name

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            # selects a value from the first dimension
            item = (item,)
        dims = tuple(
            dim for idx, dim in enumerate(self.dims)
            if idx >= len(item) or not isinstance(item[idx], int)
        )
        coords = {}
        for idx, dim in enumerate(self.dims):
            if dim not in self._coords:
                continue
            if idx >= len(item):
                coords[dim] = self._coords[dim]
                continue
            if isinstance(item[idx], int):
                continue
            coords[dim] = self._coords[dim][item[idx]]
        output = self._data[item]
        if isinstance(output, (zarr.Array, np.ndarray)):
            return DataArray(
                output,
                coords=coords,
                dims=dims,
                name=self.name,
            )
        return output

    def append(self, value: 'DataArray', axis: typing.Union[str, int] = 0):
        """Append a value to the end of the array.

        Parameters
        ----------
        value : DataArray
            The value to append.
        """
        if self.dims != value.dims:
            raise ValueError(
                f'cannot append value with dims {value.dims} '
                f'to array with dims {self.dims}'
            )
        # merge the coordinates along the axis
        if not isinstance(axis, int):
            axis = self.dims.index(axis)
        # get the name of the axis
        dim = self.dims[axis]
        if dim not in self.coords:
            raise ValueError(
                f'cannot append value with dims {value.dims} '
                f'to array with dims {self.dims} '
            )
        if dim not in value.coords:
            raise ValueError(
                f'cannot append value with dims {value.dims} '
                f'to array with dims {self.dims} '
            )
        self._data = self._data.append(value._data, axis=axis)
        new_dim_index = np.vstack([self.coords[dim], value.coords[dim]])
        self._coords._set_key_value(dim, new_dim_index)

    def shape(self):
        """Return the shape of the data array."""
        return self._data.shape


class RecordBatch(object):
    """A RecordBatch is a collection of equal-length Zarr arrays
    or other RecordBatches.
    """

    def __init__(self, store: RT, overwrite=True) -> None:
        if not isinstance(store, zarr.Group):
            store = zarr.group(store=store, overwrite=overwrite)
        self._root = store

    def __getitem__(self, item: str) -> RT:
        output = self._root[item]
        return

    def from_arrays(
            self,
            arrays_or_groups: typing.Sequence[RT],
            names: typing.Sequence[str] = None,
            store: typing.Union[str, zarr.storage.Storage] = None,
            overwrite: bool = True,
    ) -> 'RecordBatch':
        """Create a RecordBatch from a sequence of arrays.

        Parameters
        ----------
        arrays_or_groups : typing.Sequence[typing.Any]
            A sequence of arrays.
        names : typing.Sequence[str]
            A sequence of names for the arrays.

        Returns
        -------
        RecordBatch
            A RecordBatch.
        """
        rb = RecordBatch(store=store, overwrite=overwrite)
        for name, array in zip(names, arrays_or_groups):
            rb[name] = array
        return rb
