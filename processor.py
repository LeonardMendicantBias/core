# %%

# import h5py

import encodings
import enum
from operator import le
import os
from posixpath import split
from unittest import result
from xml.dom.minidom import Element
import h5py
import numpy as np
import pandas as pd

from dataclasses import InitVar, dataclass, asdict, field
from random import sample
from typing import List, Tuple, Dict
import typing
from torch.utils.data import Dataset, DataLoader

from collections import Counter, OrderedDict
from collections.abc import Iterable


@dataclass
class DataList:
    inp: List[np.ndarray]
    out: List[np.ndarray]
    split: str
    name: str
    lengths: Tuple[int, int]

    def __post_init__(self):
        assert (i:=len(self.inp)) == (j:=len(self.out)), f"sizes should be equal but {i} and {j}"

    def __len__(self):
        return len(self.inp)
        
    # write instance attributes under 'split' group under 'name' group
    def to_dataset(self, parent_group: h5py.Group):
        split = parent_group.require_group(self.split)
        group = split.require_group(self.name)

        group.attrs['name'] = self.name
        group.attrs['split'] = self.split
        group.attrs['lengths'] = np.array(self.lengths)

        group.create_dataset('inp', data=self.inp)
        group.create_dataset('out', data=self.out)

    # receive a 'name' group and parse as a DataList
    @classmethod
    def from_dataset(cls, group: h5py.Group):
        data_list = cls(
            inp=group['inp'],
            out=group['out'],
            split=group.attrs['split'],
            name=group.attrs['name'],
            lengths=tuple(group.attrs['lengths'])
        )
        return data_list


@dataclass
class Vocabulary:
    
    bos_token: str
    eos_token: str
    sep_token: str
    pad_token: str
    unk_token: str

    word_counter: InitVar[Counter] = None

    w2i: Dict = field(init=False, repr=False, default_factory=dict)
    i2w: Dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self, word_counter):
        self._special_tokens = [
            self.bos_token, self.eos_token,
            self.pad_token,
            self.pad_token, self.unk_token
        ]

        self.w2i = {token: i for i, token in enumerate(self._special_tokens)}

        if word_counter is not None:
            for key, _ in word_counter.items():
                if key not in self.w2i:
                    self.w2i[key] = len(self.w2i)

        self.i2w = {self.w2i[k]:k for k in self.w2i}

    def __len__(self):
        return len(self.w2i)

    @property
    def bos_idx(self) -> int:
        return self.w2i[self.bos_token]

    @property
    def eos_idx(self) -> int:
        return self.w2i[self.eos_token]

    @property
    def sep_idx(self) -> int:
        return self.w2i[self.sep_token]

    @property
    def pad_idx(self) -> int:
        return self.w2i[self.pad_token]

    @property
    def unk_idx(self) -> int:
        return self.w2i[self.unk_token]
        
    # @classmethod
    # def build_from_dict(cls, vocab: dict):
    #     self.w2i = vocab
    #     self.i2w = {self.w2i[k]:k for k in self.w2i}
        
    #     return cls(
    #         bos
    #     )

    # # write instance attributes under 'split' group under 'name' group
    # def to_dataset(self, parent_group: h5py.Group):
    #     split = parent_group.require_group('vocab')
    #     group = split.require_group(self.name)

    #     group.attrs['name'] = self.name
    #     group.attrs['split'] = self.split
    #     group.attrs['lengths'] = np.array(self.lengths)

    #     group.create_dataset('inp', data=self.inp)
    #     group.create_dataset('out', data=self.out)

    def word_to_index(self, words):
        return [self.w2i.get(word, self.w2i.get(self.unk_token)) for word in words]

    def index_to_word(self, idx):
        return [self.i2w.get(index, self.unk_token) for index in idx]


@dataclass
class ProcessDataList:
    inp: List[str]
    out: List[str]
    split: str
    name: str
    lengths: Tuple[int, int]
    ####
    is_share_emb: InitVar[bool] = False
    vocab_dict: Dict[Vocabulary, Vocabulary] = field(
        init=False, repr=False,
        default_factory=lambda: {'inp': None, 'out': None}
    )
    train_vocab_dict: InitVar[Dict[Vocabulary, Vocabulary]] = None

    def __post_init__(self, is_share_emb, train_vocab_dict):
        assert (i:=len(self.inp)) == (j:=len(self.out)), f"sizes should be equal but {i} and {j}"
        counter_dict = {}
        if self.split == 'train':
            counter_dict['inp'] = self._count(self.inp)
            counter_dict['out'] = self._count(self.out)
            
            if is_share_emb:
                share_counter = counter_dict['inp'] + counter_dict['out']
                counter_dict['inp'] = share_counter
                counter_dict['out'] = share_counter
            
            for key in counter_dict:
                self.vocab_dict[key] = Vocabulary(
                    '[BOS]', '[EOS]', '[PAD]', '[PAD]', '[UNK]',
                    dict(sorted(counter_dict[key].items()))
                )
            print(self.vocab_dict[key].w2i)
        else:
            assert train_vocab_dict is not None, 'test split should inherit train split vocab'
            self.vocab_dict = train_vocab_dict

        self.inp = [self.process(element, self.vocab_dict['inp']) for element in self.inp]
        self.out = [self.process(element, self.vocab_dict['out']) for element in self.out]
    
    def process(self, elements, vocab: Vocabulary):
        s = [vocab.bos_token]
        for i, element in enumerate(elements):
            s += [e for e in str(element)]
            if i != len(elements)-1:
                s+= [vocab.sep_token]
        s += [vocab.eos_token]
        return np.array(vocab.word_to_index(s), dtype=np.int32)

    def __len__(self):
        return len(self.inp)

    def _count(self, elements):
        counter = Counter()
        for element in elements:
            for e in element:
                for s in str(e):
                    counter.update(s)
        return counter

    def to_dataset(self, parent_group: h5py.Group):
        split = parent_group.require_group(self.split)
        group = split.require_group(self.name)

        group.attrs['name'] = self.name
        group.attrs['split'] = self.split
        group.attrs['lengths'] = np.array(self.lengths)

        group.create_dataset('inp', data=self.inp)
        group.create_dataset('out', data=self.out)
        
    @classmethod
    def from_datalist(cls, data_list: DataList, is_share_emb=False):
        return cls(
            inp=data_list.inp,
            out=data_list.out,
            split=data_list.split,
            name=data_list.name,
            lengths=data_list.lengths,
            is_share_emb=is_share_emb
        )


class H5Dataset(Dataset):

    def __init__(self, file_path, split='train'):
        self.file_path = file_path
        self.dataset = None
        self.split = split

        with h5py.File(file_path, "r") as f:
            self.size = f['process'][split].attrs['size']

    def __del__(self):
        if self.dataset is not None:
            self.dataset.close()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = self.dataset = h5py.File(self.file_path, 'r')#["dataset"]
        data = self.dataset['process'][self.split]

        return {data[name][idx] for name in data}


class Processor:

    def __init__(self, 
        bos_token, eos_token,
        pad_token, unk_token,
        is_share_emb: bool=True
    ):
        self.bos_token, self.eos_token = bos_token, eos_token
        self.pad_token, self.unk_token = pad_token, unk_token
        self._special_tokens = [
            self.bos_token, self.eos_token,
            self.pad_token, self.unk_token
        ]
        self.is_share_emb = is_share_emb

    def _tokenize(self, inp):
        return [s for s in str(inp)]

    def _abc(self, elements, is_appd=False):
        s = []
        for i, element in enumerate(elements):
            s += [e for e in str(element)]
            if i != len(elements)-1 and is_appd:
                s+= ['[SEP]']
        if is_appd:
            s = ['[BOS]'] + s + ['[EOS]']
        return s

    def process(self, file_path):
        with h5py.File(file_path, "r+") as f:
            raw_data_group = f['raw']
            process_data_group = f.create_group('process')
            process_train_data_group = process_data_group.create_group('train')
            process_train_data_group.attrs['split'] = 'train'

            # preprocess
            train_data_group = raw_data_group['train']
            if self.is_share_emb:
                c = Counter()
                word_counter_dict = {name: c for name in train_data_group}
            else:
                word_counter_dict = {name: Counter() for name in train_data_group}
            for name in train_data_group:
                for elements in train_data_group[name]:
                    s = self._abc(elements)
                    word_counter_dict[name].update(s)

            vocab_dict = {}
            vocab_group = process_train_data_group.create_group('vocab')
            for name in train_data_group:
                vocab = Vocabulary(
                    self.bos_token, self.eos_token, self.pad_token, self.unk_token
                )
                vocab.build_vocab(
                    OrderedDict(sorted(word_counter_dict[name].items()))
                )
                abc = vocab_group.create_group(name)
                abc.create_dataset('word',
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    data=list(vocab.w2i.keys())
                )
                abc.create_dataset('index',
                    dtype=np.dtype('int32'),
                    data=list(vocab.w2i.values())
                )

                vocab_dict[name] = vocab
                    
            # process_train_data_group.attrs['lengths'] = train_data_group.attrs['lengths']
            for name in train_data_group:
                # create h5-dataset for processed data
                ds = process_train_data_group.create_dataset(
                    name,
                    shape=len(train_data_group[name]),
                    dtype=h5py.vlen_dtype(np.dtype('int32'))
                )

                for i, elements in enumerate(train_data_group[name]):
                    s = self._abc(elements, True)
                    ds[i] = vocab.word_to_index(s)

            test_data_group = raw_data_group['test']
            for name in test_data_group:
                process_test_data_group = process_data_group.create_group(f'test-{name}')
                process_test_data_group.attrs['lengths'] = test_data_group[name].attrs['lengths']
                process_test_data_group.attrs['split'] = 'test'
                d_group = test_data_group[name]
                for n in d_group:
                    ds = process_test_data_group.create_dataset(
                        n,
                        shape=len(d_group[n]),
                        dtype=h5py.vlen_dtype(np.dtype('int32'))
                    )

                    for i, elements in enumerate(d_group[n]):
                        s = self._abc(elements, True)
                        ds[i] = vocab_dict[n].word_to_index(s)


            # # metadata of dataset
            # data_group.attrs['size'] = len(raw_data_group[name])

    def read(self, file_path):
        vocab_dict = {}
        with h5py.File('./data.h5', "r") as f:
            data = f['data']
            vocab = data['vocab']

            for name in data['vocab']:
                enc_type = h5py.check_string_dtype(vocab[name]['word'].dtype).encoding
                words = {
                    k.decode(enc_type): v for k, v in zip(vocab[name]['word'], vocab[name]['index'])
                }
                
                # vocab_dict[name] = 
                v = Vocabulary(
                    self.bos_token, self.eos_token, self.pad_token, self.unk_token
                )
                v.build_from_dict(words)
                vocab_dict[name] = v

        dataset = H5Dataset(file_path)

        return dataset, vocab_dict


def _gen_add_neg(size: int, length: Tuple[int, int], neg_prob: float=0.):
    a = np.random.randint(10**length[0], 10**length[1], size=size)
    is_neg = np.random.choice([1, -1], size, p=[1-neg_prob, neg_prob])

    return a * is_neg


def gen_data(size: int, length: Tuple[int, int], neg_prob: float=0.):
    a = _gen_add_neg(size, length, neg_prob)
    b = _gen_add_neg(size, length, neg_prob)
    
    inp = np.stack([a, b], axis=-1)
    out = (a+b)[..., np.newaxis]

    return inp, out
    


if __name__ == '__main__':
    # RAW data
    ds_size, length, neg_prob = 10, (0, 2), 0.25

    # save raw dataset to h5
    with h5py.File("./data.h5", "w") as fp:
        fp.attrs['name'] = 'add'

        raw_group = fp.create_group('raw')

        train_data = DataList(
            *gen_data(ds_size, length, neg_prob), 
            'train', f'{length[0]}-{length[1]}', length
        )
        train_data.to_dataset(raw_group)

        for l in [3, 4]:
            test_data = DataList(
                *gen_data(ds_size, length, neg_prob), 
                'test', f'{l-1}-{l}', (l-1, l)
            )
            test_data.to_dataset(raw_group)

    # processor = Processor(
    #     '[BOS]', '[EOS]', '[PAD]', '[UNK]'
    # )
    # processor.process("./data.h5")

    # processor = Processor(
    #     '[BOS]', '[EOS]', '[PAD]', '[UNK]'
    # )
    # processor.read("./data.h5")

    with h5py.File('./data.h5', "r") as f:
        raw = f['raw']
        # for name in raw:
        group = raw['train']
        for n, h5obj in group.items():
            data_list = DataList.from_dataset(h5obj)
            process_data_list = ProcessDataList.from_datalist(data_list, is_share_emb=True)
            break
            # break
                

# %%

if __name__ == '__main__':
    with h5py.File('./data.h5', "r") as f:
        raw = f['raw']
        for name in raw:
            for n in raw[name]:
                print(name, n)
        
        # for name in vocab_group:
        #     print(
        #         type(vocab_group[name][0].decode('utf-8'))
        #     )
        #     break

# %%

if __name__ == '__main__':
    ds = H5Dataset('/mount/viz/processed.h5')
    loader = DataLoader(ds, batch_size=2, pin_memory=True, num_workers=2)
    for batch in loader:
        print(batch)

    del loader
    del ds
# %%

if __name__ == '__main__':
    length, size = 2, 10
    
    a = np.random.randint(10**length, size=(size, 1))
    b = np.random.randint(10**length, size=(size, 1))
    print(np.concatenate([a, b], axis=-1))
    print((a+b))
