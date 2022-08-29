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


class H5Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = None
        # self.operands = self.hdf5_file['data']['operands']
        # self.result = self.hdf5_file['data']['result']

        with h5py.File(file_path, "r") as f:
            self.length = f.attrs['length']

    def __del__(self):
        if self.dataset is not None:
            self.dataset.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = self.dataset = h5py.File(self.file_path, 'r')#["dataset"]
        data = self.dataset['data']

        return {
            'operands': data['operands'][idx],
            'result': data['result'][idx]
        }


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


class Processor:

    def __init__(self, 
        bos_token, eos_token,
        pad_token, unk_token,
        is_shared_vocab: bool=True
    ):
        self.bos_token, self.eos_token = bos_token, eos_token
        self.pad_token, self.unk_token = pad_token, unk_token
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
        # 2: BOS and EOS token
        # number of SEP token between elements
        s += ['[PAD]']*(max_len + 2 + (len(elements)-1) - len(s))
        return s

    def _process(self,
        data_list: Iterable,
        max_len: int=None,
        word_counter: Counter=None
    ) -> Vocabulary:
        max_len = max_len or self._max_in_list(data_list)
        word_counter = word_counter or Counter()
        
        for elements in data_list:
            s = self._abc(elements, max_len)
            
            # update vocabulary
            word_counter.update(s)

        return word_counter, max_len

    def process(self, file_path):
        with h5py.File(file_path, "r") as f, \
            h5py.File("/mount/viz/processed.h5", "w") as pf:

            raw_data_group = f['data']
            pdf_data_group = pf.create_group('data')

            # preprocess
            word_counter_dict, max_len_dict = {}, {}
            for name in raw_data_group:
                word_counter, max_len = self._process(raw_data_group[name])
                word_counter_dict[name] = word_counter
                max_len_dict[name] = max_len
                # print(name, max_len)

            # build vocabulary
            vocab_dict = {}
            if self.is_shared_vocab:
                # sum has a start object (int=0 as default)
                # -> initialize an empty Counter to avoid error
                word_counter = sum(word_counter_dict.values(), Counter())
                vocab = Vocabulary(
                    self.bos_token, self.eos_token, self.pad_token, self.unk_token
                )
                vocab.build_vocab(
                    OrderedDict(sorted(word_counter.items()))
                )
                vocab_dict['share'] = vocab
            else:
                for name in raw_data_group:
                    vocab = Vocabulary(
                        self.bos_token, self.eos_token, self.pad_token, self.unk_token
                    )
                    vocab.build_vocab(
                        OrderedDict(sorted(word_counter_dict[name].items()))
                    )
                    vocab_dict[name] = vocab
            
            # write processed data
            for name in raw_data_group:
                vocab = vocab_dict['share'] if self.is_shared_vocab else vocab_dict[name]

                # create h5-dataset for processed data
                ds = pdf_data_group.create_dataset(
                    name,
                    shape=(
                        len(raw_data_group[name]),
                        max_len_dict[name]+ 2 + raw_data_group[name].shape[-1]-1
                    ),
                    # length of sequences + (BOS, EOS) + PAD
                    dtype=np.int32
                )

                for i, elements in enumerate(raw_data_group[name]):
                    s = self._abc(elements, max_len_dict[name])
                    s = np.array(vocab.word_to_index(s), dtype=np.int32)
                    ds[i] = s

            vocab_group = pf.create_group('vocab')
            if self.is_shared_vocab:
                vocab_group.create_dataset('share_keys',
                    dtype=h5py.special_dtype(vlen=str),
                    data=list(vocab_dict['share'].w2i.keys())
                )
                vocab_group.create_dataset('share_values',
                    data=list(vocab_dict['share'].w2i.values())
                )
            else:
                for name in raw_data_group:
                    vocab_group.create_dataset(f'{name}_keys',
                        dtype=h5py.special_dtype(vlen=str),
                        data=list(vocab_dict[name].w2i.keys())
                    )
                    vocab_group.create_dataset(f'{name}_values', data=list(vocab_dict[name].i2w.values()))

            pf.attrs['name'] = f.attrs['name']
            pf.attrs['length'] = len(raw_data_group[name])


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

    processor = Processor(
        '[BOS]', '[EOS]', '[PAD]', '[UNK]'
    )
    processor.process("./raw_data.h5")

# %%

with h5py.File('./processed.h5', "r") as f:
    vocab_group = f['vocab']
    for name in vocab_group:
        print(
            type(vocab_group[name][0].decode('utf-8'))
        )
        break

# %%

if __name__ == '__main__':
    ds = H5Dataset('/mount/viz/processed.h5')
    loader = DataLoader(ds, batch_size=2, pin_memory=True, num_workers=2)
    for batch in loader:
        print(batch)

    del loader
    del ds
# %%
