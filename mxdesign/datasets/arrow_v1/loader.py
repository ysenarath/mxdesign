"""Defines base-classes for types of datasets."""
import functools
from typing import Any
import typing
from urllib.parse import urlparse

import pyarrow as pa
from mxdesign.datasets.dataset import Dataset, DatasetInfo
from mxdesign.datasets.helpers import HFDatasetIterator

try:
    # if datasets is installed, import it
    import datasets as hfd
except ImportError:
    hfd = None

__all__ = [
    'get_dataset_loader',
    'register',
]


datasets = {}
dataset_aliases = {}


class DatasetLoader(object):
    def __init__(self, func, /, **kwargs: Any) -> None:
        self._loader: typing.Callable = func
        self.info = DatasetInfo(**kwargs)

    def load(self, *args: Any, **kwds: Any) -> Dataset:
        obj = self._loader(*args, **kwds)
        return Dataset(obj, info=self.info)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.load(*args, **kwds)


def register(name, aliases=None, splits=None):
    def decorator(func: typing.Callable):
        datasets[name] = DatasetLoader(
            func,
            name=name,
            aliases=aliases,
            splits=splits,
        )
        dataset_aliases[name] = name
        if aliases is not None:
            for alias in aliases:
                dataset_aliases[alias] = name
        return func
    return decorator


def hf_dataset_loader(path, *args, **kwargs):
    """Load a huggingface dataset."""
    dsd = hfd.load_dataset(path, *args, **kwargs)
    if isinstance(dsd, (hfd.Dataset, hfd.DatasetDict)):
        return HFDatasetIterator(dsd)
    raise TypeError(
        f'unsupported dataset type: {type(dsd).__name__}'
    )


def get_dataset_loader(*args, **kwargs):
    """Get dataset loader."""
    if len(args) < 1:
        raise ValueError(
            'get_dataset_loader() takes at least 1 argument (0 given)'
        )
    path_or_name = args[0]
    args = args[1:]
    uri = urlparse(path_or_name)
    # create a new dynamic dataset
    uri_path = uri.path
    if isinstance(uri_path, bytes):
        uri_path = uri_path.decode('utf-8')
    # strip trailing slash
    uri_path = uri_path.rstrip('/')
    if uri.scheme == 'file' and (
        uri.path.endswith('.jsonl') or
        uri.path.endswith('.ndjson')
    ):
        if uri.hostname is not None:
            # prepend hostname to path
            uri_path = uri.hostname + uri_path
        splits = tuple()
        return DatasetLoader(
            functools.partial(pa.json.read_json, uri_path),
            splits=splits
        )
    if uri.scheme == 'file' and uri.path.endswith('.csv'):
        if uri.hostname is not None:
            uri_path = uri.hostname + uri_path
        splits = tuple()
        return DatasetLoader(
            functools.partial(pa.csv.read_csv, uri_path),
            splits=splits
        )
    if (uri.scheme in ('http', 'https') and uri.hostname == 'huggingface.co') \
            or uri.scheme == 'hf':
        # remove prefix '/datasets/' from url path
        # len('/datasets/') == 10
        if uri.scheme != 'hf':
            assert uri_path.startswith('/datasets/'), \
                f'path must start with \'/datasets/\', found \'{uri_path}\''
            uri_path = uri_path[10:]
        else:
            # remove prefix '/'
            uri_path = uri_path.lstrip('/')
        # create a new dynamic huggingface dataset
        ds_builder = hfd.load_dataset_builder(
            uri_path, *args, **kwargs
        )
        splits = tuple()
        if ds_builder.info.splits is not None:
            splits = tuple(ds_builder.info.splits.keys())
        return DatasetLoader(
            functools.partial(hf_dataset_loader, uri_path),
            splits=splits
        )
    alias = dataset_aliases[uri.path]
    return datasets[alias]
