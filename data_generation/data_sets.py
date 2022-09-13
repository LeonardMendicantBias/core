# %%

from pathlib import Path
import h5py
from dataclasses import dataclass, InitVar
from typing import List, Tuple
import numpy as np

from base import SplitType, DataSplit, ProcessDataSplit
import torch
from torch.utils.data import Dataset


def _gen_add_neg(size: int, lengths: Tuple[int, int], neg_prob: float=0.):
    a = np.random.randint(10**(lengths[0]-1), 10**lengths[1], size=size)
    is_neg = np.random.choice([1, -1], size, p=[1-neg_prob, neg_prob])

    return a * is_neg


def gen_data(size: int, lengths: Tuple[int, int], neg_prob: float=0.):
    a = _gen_add_neg(size, lengths, neg_prob)
    b = _gen_add_neg(size, lengths, neg_prob)
    
    inp = np.stack([a, b], axis=-1)
    out = inp.sum(-1, keepdims=True)

    return inp, out


class H5Dataset(Dataset):

    def __init__(self, dataset, split: SplitType, name:str):
        self.dataset = None
        self.split = split
        self.name = name
        self.file_path = dataset

        with h5py.File(f"/mount/dataset/{dataset}", "r") as f:
            self.size = f['process'][split.name][name].attrs['size']

    def __del__(self):
        if self.dataset is not None:
            self.dataset.close()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(f"/mount/dataset/{self.file_path}", 'r')
        data = self.dataset['process'][self.split.name][self.name]

        return {name: data[name][idx] for name in data}

    @classmethod
    def from_process_data(cls, data_split: ProcessDataSplit):
        return cls('ADD.h5', data_split.split, data_split.name)


class BaseDataset:
    '''
        A set of splits for training, validating, and testing
        Read the splits from disk, otherwise generate then save
        Note: The save data is human-readable (not PyTorch)
    '''

    def __init__(self,
        name: str,
        train_split: ProcessDataSplit,
        val_splits: List[ProcessDataSplit],
        test_splits: List[ProcessDataSplit],
    ):
        self.name = name
        self.train_split = train_split
        self.val_splits = val_splits
        self.test_splits = test_splits

        self.train_set = H5Dataset.from_process_data(train_split)
        self.val_sets = [H5Dataset.from_process_data(split) for split in val_splits]
        self.test_sets = [H5Dataset.from_process_data(split) for split in test_splits]

    # def to_disk(self, cache_dir):
    #     with h5py.File(cache_dir, "w") as fp:
    #         fp.attrs['name'] = self.name

    #         self.train_split.to_dataset(fp)
    #         [split.to_dataset(fp) for split in self.val_splits]
    #         [split.to_dataset(fp) for split in self.test_splits]

    # @classmethod
    # def from_disk(cls, file_path):
    #     with h5py.File(file_path, "r") as fp:
    #         name = fp.attrs['name']

    #         train_split = ProcessDataSplit.from_dataset(fp, SplitType.TRAIN)
    #         val_splits = ProcessDataSplit.from_dataset(fp, SplitType.VAL)
    #         test_splits = ProcessDataSplit.from_dataset(fp, SplitType.TEST)

    #     return cls(name, train_split, val_splits, test_splits)
        
    @classmethod
    def from_raw(cls,
        name,
        train_split: DataSplit,
        val_splits: List[DataSplit],
        test_splits: List[DataSplit],
    ):
        with h5py.File(f'/mount/dataset/{name}.h5', "w") as fp:
            # process raw data
            train_split = ProcessDataSplit.from_datalist(
                train_split, is_share_emb=True,
            )
            train_split.to_dataset(fp)

            vocab_dict = train_split.vocab_dict
            val_splits = []
            for split in val_splits:
                split = ProcessDataSplit.from_datalist(
                    split, is_share_emb=True, vocab_dict=vocab_dict
                )
                split.to_dataset(fp)
                val_splits.append(split)
            test_splits = []
            for split in test_splits:
                split = ProcessDataSplit.from_datalist(
                    split, is_share_emb=True, vocab_dict=vocab_dict
                )
                split.to_dataset(fp)
                test_splits.append(split)
        
        return cls(name, train_split, val_splits, test_splits)


if __name__ == '__main__':
    datase = BaseDataset.from_disk(
        asd
    )

    for sample in datase.train_set:
        print(sample)
        break

# %%
