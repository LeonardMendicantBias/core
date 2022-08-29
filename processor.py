# %%

# import h5py

import encodings
import enum
import os
from unittest import result
from xml.dom.minidom import Element
import h5py
import numpy as np
import pandas as pd

from dataclasses import dataclass, asdict
from random import sample
from typing import List
from torch.utils.data import Dataset, DataLoader

from collections import Counter, OrderedDict
from collections.abc import Iterable

@dataclass(frozen=True)
class RawSample:
    operands: List[List[str]]
    result: List[str]


@dataclass(frozen=True)
class Seq:
    seq: List[str]

@dataclass(frozen=True)
class RawSeqs2Seqs:
    operands: List[Seq]
    result: List[Seq]
    

class H5Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_path):
        with h5py.File(file_path, "r") as f:
            self.dataset_name = f.attrs['name']
            self.data_group = f['data']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        operands = sample.get('operands')
        result = sample.get('result')

        return operands, result


class Processor:

    class OrderedCounter(Counter, OrderedDict):
        pass

    class Vocabulary:
        
        def __init__(self,
            bos_token, eos_token,
            pad_token, unk_token
        ):
            self.bos_token, self.eos_token = bos_token, eos_token
            self.pad_token, self.unk_token = pad_token, unk_token
            self._special_tokens = [
                self.bos_token, self.eos_token,
                self.pad_token, self.unk_token
            ]

            self.w2i, self.i2w = None, None

        def build_vocab(self, word_counter: Counter):
            self.w2i = OrderedDict({token: i for i, token in enumerate(self._special_tokens)})
            self.i2w = OrderedDict()

            for key, value in word_counter.items():
                if key not in self.w2i:
                    self.w2i[key] = len(self.w2i)

            self.i2w = {self.w2i[k]:k for k in self.w2i}

        def word_to_index(self, words):
            return [self.w2i.get(word, self.w2i.get(self.unk_token)) for word in words]

        def index_to_word(self, idx):
            return [self.i2w.get(index, self.unk_token) for index in idx]

    def __init__(self, 
        is_shared_vocab: bool=True
    ):
        self.is_shared_vocab = is_shared_vocab

    def _tokenize(self, inp):
        return str(inp)

    # for each row in dataset:
    #   for each sample in row:
    def _max_in_list(self, l) -> int:
        return max(
            map(
                lambda samples: sum([
                    len(self._tokenize(sample)) for sample in samples
                ]), l
            )
        )

    def _abc(self, elements, max_len):
        s = ['[BOS]']
        for i, element in enumerate(elements):
            s += [e for e in str(element)]
            if i != len(elements)-1:
                s+= ['[SEP]']
        s += ['[EOS]']
        
        # compensate [PAD] to maximum lenth
        s += ['[PAD]']*(max_len + 2 + (len(str(element))-1) - len(s))
        return s

    def _process(self,
        raw_group: Iterable,
        pro_group,
        name,
        max_len: int=None,
        word_counter: OrderedCounter=None
    ) -> Dataset:
        data_list = raw_group[name]
        max_len = max_len or self._max_in_list(data_list)
        # word_counter = word_counter or self.OrderedCounter()
        word_counter = word_counter or Counter()
        
        pro_group.create_dataset(
            name,
            (len(data_list), max_len),
            dtype=np.int32
        )

        for elements in data_list:
            # append [BOS] and [EOS] to tokenized 
            s = self._abc(elements, max_len)
            
            # update vocabulary
            word_counter.update(s)

            # convert token to index
            
        vocab = Processor.Vocabulary(
            '[BOS]', '[EOS]', '[PAD]', '[UNK]'
        )
        vocab.build_vocab(
            OrderedDict(sorted(word_counter.items()))
        )

        return OrderedDict(sorted(word_counter.items()))

    def process(self, file_path):
        with h5py.File(file_path, "r") as f, \
            h5py.File("./processed.h5", "w") as pf:
            dataset_name = f.attrs['name']

            raw_data_group = f['data']
            pdf_data_group = pf.create_group('data')

            for name in raw_data_group:
                # print(name)
                word_counter = self._process(raw_data_group, pdf_data_group, name)

                break
            # print(word_counter)
            # print(type(word_counter.keys()))
            # self._process(data_group['result'], word_counter=word_counter)
            # print(word_counter)


def _gen(size, length, neg_prob=0.):
    a = np.random.randint(10**length, size=size)
    is_neg = np.random.choice([1, -1], size, p=[1-neg_prob, neg_prob])
    return a * is_neg


if __name__ == '__main__':
    # RAW data
    a = _gen(10, 2, 0.25)
    b = _gen(10, 2, 0.25)

    # save raw dataset to h5
    # with h5py.File("./raw_data.h5", "w") as fp:
    #     fp.attrs['name'] = 'add'

    #     data_group = fp.create_group('data')
    #     data_group.attrs['size'] = a.shape[0]
    #     data_group.attrs['length'] = 2

    #     data_group.create_dataset(
    #         'operands',
    #         shape=(10, 2),
    #         data=np.stack([a, b], axis=-1)
    #     )
    #     data_group.create_dataset(
    #         'result',
    #         shape=(10, 1),
    #         data=a + b
    #     )

    processor = Processor()
    processor.process("./raw_data.h5")

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

print(
    np.array(
        [i for i in str(123)],
        dtype='S10'
    ).dtype
)

# %%

# with h5py.File('./raw_data.h5', "r") as f:
#     print(f.attrs.keys())
#     print(f['data']['operands'][()])
#     print(f['data']['result'][()])
    # for name in f['df']:
    #     # print(name)
    #     print(f['df'][name][()])
    #     print('-'*30)

# %%

# df = pd.read_hdf('./data.h5', 'df')  

# %%

l = [123, 1234]

arr = np.array(
    [np.array([i for i in str(b)]) for b in l],
    dtype=object
)
arr
# %%
