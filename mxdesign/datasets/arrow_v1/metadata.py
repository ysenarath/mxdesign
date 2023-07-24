from collections.abc import Mapping
import json
import typing
import weakref

from pyarrow import dataset as ds

Dataset = typing.TypeVar('Dataset')

__all__ = [
    'Metadata'
]


class Metadata(Mapping):
    """A wrapper around the dataset metadata.

    Notes
    -----
    The metadata is stored as a dictionary of bytes. This class converts the
    keys to strings when getting and setting items. The values loaded using
    `json.loads` function.

    This class is a wrapper around the dataset metadata and should not be used
    to store large amounts of data.
    """

    def __init__(self, dataset: Dataset) -> None:
        self._dataset: Dataset = weakref.proxy(dataset)

    @property
    def _base_dataset(self) -> ds.Dataset:
        """Return the dataset this metadata belongs to."""
        return self._dataset._base_dataset

    @_base_dataset.setter
    def _base_dataset(self, value: ds.Dataset) -> typing.Any:
        """Set the dataset this metadata belongs to."""
        self._dataset._base_dataset = value

    def __iter__(self) -> typing.Iterator[str]:
        """Return an iterator over the metadata keys."""
        if self._base_dataset.schema.metadata is None:
            raise StopIteration
        yield from map(
            lambda x: x.decode('utf-8'),
            self._base_dataset.schema.metadata.keys(),
        )

    def __len__(self) -> int:
        """Return the number of metadata keys."""
        if self._base_dataset.schema.metadata is None:
            return 0
        return len(self._base_dataset.schema.metadata)

    def __getitem__(self, key: str) -> typing.Any:
        """Return the metadata value for the given key."""
        if self._base_dataset.schema.metadata is None:
            raise KeyError(key)
        value = self._base_dataset.schema.metadata[key.encode('utf-8')]
        return json.loads(value)

    def __setitem__(self, key: str, value: typing.Any) -> None:
        """Set the metadata value for the given key."""
        metadata = self._base_dataset.schema.metadata or {}
        metadata[key] = json.dumps(value)
        schema = self._base_dataset.schema.with_metadata(metadata)
        self._base_dataset = self._base_dataset.replace_schema(schema)
