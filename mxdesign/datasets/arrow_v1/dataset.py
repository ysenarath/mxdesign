from dataclasses import field
from pathlib import Path
import tempfile
import typing
import pandas as pd

import pyarrow as pa
from pyarrow import dataset as ds
from mxdesign.callbacks.progbar import ProgbarLogger

from mxdesign.core.schema import Schema
from mxdesign.core.version import Version
from mxdesign.datasets.metadata import Metadata
from mxdesign.datasets.helpers import BatchIterator, create_batches
from mxdesign.utils import cache

__all__ = [
    'Metadata',
    'Dataset',
    'DatasetInfo',
    'Table',
]

TD = typing.Optional[tempfile.TemporaryDirectory]

Table = pa.Table


class DatasetInfo(Schema):
    name: typing.Optional[str] = None
    description: typing.Optional[str] = None
    version: Version = Version('0.0.1')
    citation: typing.Optional[str] = None
    license: typing.Optional[str] = None
    splits: typing.List[str] = field(default_factory=list)
    aliases: typing.List[str] = field(default_factory=list)


class Dataset(object):
    def __init__(self, obj, info: DatasetInfo = None, **kwargs):
        """Create a new dataset.

        Parameters
        ----------
        obj : Iterable
            The iterable to convert to a dataset.
        info : DatasetInfo, optional
            The dataset info.

        Other Parameters
        ----------------
        format : str, optional
            The format to use when writing the dataset to disk.
        **kwargs
            Additional keyword arguments to pass to the dataset writer.

        Notes
        -----
        The dataset is written to a temporary directory and deleted when the
        dataset is garbage collected.
        """
        if isinstance(obj, ds.Dataset):
            self._base_dataset: ds.Dataset = obj
            self._cachedir: TD = None
        else:
            cachedir = cache.gettempdir(prefix='dataset-')
            self._cachedir: TD = cachedir
            fp = Path(cachedir.name)
            if not isinstance(obj, BatchIterator):
                obj = create_batches(obj, **kwargs)
            obj.callbacks.clear()
            progbar = ProgbarLogger()
            obj.callbacks.attach(progbar)
            ds.write_dataset(
                # accesses schema from batch iterator
                obj, fp, schema=obj.schema, format='arrow', **kwargs
            )
            self._base_dataset: ds.Dataset = ds.dataset(
                fp, format='arrow'
            )
        self._base_metadata: Metadata = Metadata(self)
        # set info
        if isinstance(info, DatasetInfo):
            info = info.to_dict()
        self._base_metadata['info'] = info or {}

    @property
    def info(self) -> DatasetInfo:
        """Return the dataset info."""
        info = self._base_metadata.get('info', {})
        return DatasetInfo.from_dict(info)

    def __getitem__(self, item: int) -> typing.Any:
        """Return the item at the given index."""
        if isinstance(item, int):
            doc = self._base_dataset.take([item]).to_pylist()[0]
            return doc
        raise TypeError(f'unsupported index type: {type(item).__name__}')

    def __len__(self) -> int:
        """Return the number of rows in the dataset."""
        return self._base_dataset.count_rows()

    def __iter__(self) -> typing.Generator[dict, None, None]:
        """Iterate over the dataset."""
        for batch in self._base_dataset.to_batches():
            for doc in batch.to_pylist():
                yield doc

    def map(
        self, func: typing.Callable, *,
        batched: bool = False
    ) -> BatchIterator:
        """Apply a function to each item in the dataset and return
        a new dataset.

        Parameters
        ----------
        func : Callable
            The function to apply.
        batched : bool, optional
            Whether the function takes a batch of items as input,
            by default False

        Returns
        -------
        Dataset
            The new dataset.
        """
        if not callable(func):
            raise TypeError(f'func must be callable, got {type(func)}')
        batches = create_batches(self._base_dataset) \
            .map(func, batched=batched)
        return Dataset(batches)

    def merge(self, other: 'Dataset') -> 'BatchIterator':
        """Merge two datasets together and return a new dataset.

        Notes
        -----
        This method will keep the schema of the first dataset.

        Parameters
        ----------
        other : Dataset
            The other dataset to merge with.

        Returns
        -------
        BatchIterator
            The new dataset.
        """
        other = create_batches(other._base_dataset)
        batches = create_batches(self._base_dataset) \
            .merge(other)
        return Dataset(batches)

    @classmethod
    def load_from_disk(
        cls, fp: typing.Union[str, Path], version: Version = None,
        format: str = 'arrow', **kwargs
    ) -> 'Dataset':
        """Load a dataset from disk.

        Parameters
        ----------
        fp : Union[str, Path]
            The path to the dataset.
        version : Optional[Union[str, int]], optional
            The version of the dataset to load, by default None
        format : str, optional
            The format of the dataset, by default 'parquet'
        kwargs : dict
            Additional keyword arguments to pass to the dataset loader.

        Returns
        -------
        Dataset
            The loaded dataset.
        """
        if isinstance(fp, str):
            fp = Path(fp)
        source = fp
        if version is not None:
            source = source / str(version)
        obj: ds.Dataset = ds.dataset(
            source=source, format=format, **kwargs
        )
        return cls(obj)

    def save_to_disk(
        self, fp: typing.Union[str, Path], version=None,
        format='arrow', **kwargs
    ) -> None:
        """Save the dataset to disk.

        Parameters
        ----------
        fp : Union[str, Path]
            The path to save the dataset.
        version : Optional[Union[str, int]], optional
            The version of the dataset to save, by default None
        format : str, optional
            The format of the dataset, by default 'parquet'
        kwargs : dict
            Additional keyword arguments to pass to the dataset saver.

        Returns
        -------
        None
            None.
        """
        if isinstance(fp, str):
            path = Path(fp)
        if version is not None:
            info = self.info
            info.version = Version(version)
            self.info = info
        if self.info.version is None:
            raise ValueError('version must be specified')
        create_batches(self._base_dataset).save_to_disk(
            path / str(self.info.version),
            format=format,
            **kwargs
        )

    def filter(
        self, expression: ds.Expression
    ) -> 'BatchIterator':
        """Filter the dataset and return a new dataset.

        Parameters
        ----------
        expression : Expression
            The filter expression.

        Returns
        -------
        Dataset
            The new dataset.
        """
        dset = self._base_dataset.filter(expression)
        return Dataset(dset)

    def head(self, num_rows, *args, **kwargs) -> pd.DataFrame:
        """Return the first item in the dataset."""
        return self._base_dataset.head(num_rows, *args, **kwargs).to_pandas()
