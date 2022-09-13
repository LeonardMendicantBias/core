from re import L, S
import h5py
from enum import Enum, unique
from dataclasses import dataclass, field
from typing import List
import numpy as np

from torch.utils.data import Dataset
from copy import deepcopy


class Singleton (type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class H5Dataset(Dataset):

    def __init__(self, file_path, split, name):
        self.file_path = file_path
        self.dataset = None
        self.split = split
        self.name = name

        with h5py.File(file_path, "r") as f:
            self.size = f[split][name].attrs['size']

    def __del__(self):
        if self.dataset is not None: self.dataset.close()

    def __len__(self): return self.size

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = self.dataset = h5py.File(self.file_path, 'r')#["dataset"]
        data = self.dataset[self.split][self.name]

        return {name: data[name][idx] for name in data}


@dataclass
class Data:
    data: List[np.ndarray]

    def __len__(self): return len(self.data)


@dataclass
class Split:

    @unique
    class SplitType(Enum):
        TRAIN=1
        VAL=2
        TEST=3
    
    
    inp: Data
    out: Data
    split: SplitType
    name: str

    def __post_init__(self):
        assert (i:=len(self.inp)) == (j:=len(self.out)), f"sizes should be equal but {i} and {j}"

    def __len__(self): return len(self.inp)

    @property
    def ds(self):
        return H5Dataset(
            # file_path=file_path,
            split=self.split.name,
            name=self.name,
        )

    @staticmethod
    def to_arda(ds_name, split, split_name):
        file_path = f'/mount/dataset/{ds_name}.h5'
        with h5py.File(file_path, 'r') as f:
            split = f[split.upper()]
            ds = split[split_name]
            d = dict(ds.attrs).copy()

        for name, value in d.items():
            print(name, value)
        return {
            'name': d['name'],
            'split': d['split'],
            'size': str(d['size'])
        }

    def to_disk(self, h5_file: h5py.File):
        split = h5_file.require_group(self.split.name)
        group = split.require_group(self.name)

        group.attrs['name'] = self.name
        group.attrs['split'] = self.split.name
        group.attrs['size'] = len(self)

        group.create_dataset(
            'inp',
            dtype=h5py.vlen_dtype(np.dtype('int32')),
            data=self.inp.data
        )
        group.create_dataset(
            'out',
            dtype=h5py.vlen_dtype(np.dtype('int32')),
            data=self.out.data
        )

@dataclass
class Task:
    
    name: str

    train_split: Split
    val_splits: List[Split]=field(default_factory=list)
    test_splits: List[Split]=field(default_factory=list)

    def __post_init__(self):
        self.file_path = f'/mount/dataset/{self.name}.h5'
        with h5py.File(self.file_path, 'w') as f:
            self.train_split.to_disk(f)
            [split.to_disk(f) for split in self.val_splits]
            [split.to_disk(f) for split in self.test_splits]

    @staticmethod
    def to_arda(name):
        file_path = f'/mount/dataset/{name}.h5'
        with h5py.File(file_path, 'r') as f:
            train_split = f[Task.Split.SplitType.TRAIN.name]
            train_split = list([s for s in train_split])

            # splits = f[Task.Split.SplitType.VAL]

            test_splits = f[Task.Split.SplitType.TEST.name]
            test_splits = list([s for s in test_splits])
            
        return {
            'name': name,
            'train': train_split,
            # 'val': [split.name for split in self.val_splits],
            'test': test_splits,
        }
