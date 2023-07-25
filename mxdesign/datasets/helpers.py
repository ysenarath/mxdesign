from datasets import DatasetDict

__all__ = [
    'HFDatasetIterator',
]


class HFDatasetIterator(object):
    def __init__(self, dataset) -> None:
        # : typing.Union[hfd.Dataset, hfd.DatasetDict]
        self._dataset = dataset
        self._length = None

    def _compute_length(self) -> int:
        if isinstance(self._dataset, DatasetDict):
            return sum(ds.num_rows for ds in self._dataset.values())
        return self._dataset.num_rows

    def __len__(self) -> int:
        if self._length is None:
            self._length = self._compute_length()
        return self._length

    def __iter__(self):
        if isinstance(self._dataset, DatasetDict):
            for key, hfds in self._dataset.items():
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                for item in hfds:
                    if 'split' not in item:
                        item['split'] = key
                    yield item
            return
        yield from self._dataset
