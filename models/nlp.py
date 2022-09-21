# %%
from re import T
import h5py
from dataclasses import field, dataclass, InitVar
from collections import Counter
from typing import Iterable, List, Union, Tuple, Dict
import numpy as np

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .base import Pair, Split, Result, Task, Simulation, Temp


@dataclass
class Vocabulary:
    bos_token: str
    eos_token: str
    sep_token: str
    pad_token: str
    unk_token: str
    dash_token: str
    token_counter: InitVar[Counter]

    # name: str=None

    w2i: Dict=field(repr=False, default_factory=dict)
    i2w: Dict=field(repr=False, default_factory=dict)

    def __post_init__(self, token_counter):
        # after __init__ fill in the variable
        # assert that vocabulary is initialized or buildable
        assert bool(token_counter) or (bool(self.w2i) and bool(self.i2w)), "Dictionary error"

        self._special_tokens = [
            self.bos_token, self.eos_token,
            self.sep_token,
            self.pad_token, self.unk_token, self.dash_token
        ]
        
        if bool(token_counter):
            for i, token in enumerate(self._special_tokens):
                self.w2i[token] = i
            for name in dict(sorted(token_counter.items())):  # alphabetic sort
                self.w2i[name] = len(self.w2i)
            for key, value in self.w2i.items():
                self.i2w[str(value)] = key
        else:
            assert len(self.w2i) == len(self.i2w)

    def __len__(self): return len(self.w2i)

    @property
    def bos_idx(self) -> int: return self.w2i[self.bos_token]

    @property
    def eos_idx(self) -> int: return self.w2i[self.eos_token]

    @property
    def sep_idx(self) -> int: return self.w2i[self.sep_token]

    @property
    def pad_idx(self) -> int: return self.w2i[self.pad_token]

    @property
    def unk_idx(self) -> int: return self.w2i[self.unk_token]

    @property
    def dash_idx(self) -> int: return self.w2i[self.dash_token]

    def word_to_index(self, words):
        return [self.w2i.get(w, self.unk_idx) for w in words]

    def index_to_word(self, idx):
        return [self.i2w.get(str(i), self.unk_token) for i in idx]


@dataclass
class ADDTask(Task):
    inp_vocab: Vocabulary=None
    label_vocab: Vocabulary=None

    @staticmethod
    def gen_data(size: int, lengths: Tuple[int, int], neg_prob: float=0.):

        def _gen_add_neg(size: int, n_digits: Tuple[int, int], neg_prob: float=0.):
            # 10**0 = 1
            # 10**1 = 10
            # 10**2 = 100
            # 10**3 = 1000
            # (3, 3): [100, 1000) = [100, 999]
            a = np.random.randint(10**(n_digits[0]-1), 10**n_digits[1], size=size)
            is_neg = np.random.choice([1, -1], size, p=[1.-neg_prob, neg_prob])

            return a * is_neg

        a = _gen_add_neg(size, lengths, neg_prob)
        b = _gen_add_neg(size, lengths, neg_prob)
        
        inp = np.stack([a, b], axis=-1)
        label = inp.sum(-1, keepdims=True)

        return inp, label

    @staticmethod
    def tokenize(s) -> List: return str(s)

    @staticmethod
    def _count(elements) -> Counter:
        counter = Counter()
        for sample in elements:
            for sentence in sample:
                for word in ADDTask.tokenize(sentence):
                    counter.update(word)
        return counter
    
    def __post_init__(self):
        super().__post_init__()
    
    @classmethod
    def from_config(cls, 
        train_config: Tuple[int, Tuple[int, int]],
        val_configs: List[Tuple[int, Tuple[int, int]]],
        test_configs: List[Tuple[int, Tuple[int, int]]],
        is_share_emb=True
    ):
        train_raw_data = ADDTask.gen_data(train_config[0], train_config[1], 0.25)
        kw_dict = {
            'bos_token': '[BOS]', 'eos_token': '[EOS]',
            'sep_token': '[SEP]',
            'pad_token': '[PAD]', 'unk_token': '[UNK]',
            'dash_token': '#',
        }
        inp_token_count = ADDTask._count(train_raw_data[0])
        label_token_count = ADDTask._count(train_raw_data[1])

        if is_share_emb:
            inp_vocab = label_vocab = Vocabulary(**kw_dict, token_counter=inp_token_count + label_token_count)
        else:
            inp_vocab = Vocabulary(**kw_dict, token_counter=inp_token_count)
            label_vocab = Vocabulary(**kw_dict, token_counter=label_token_count)

        test_raw_datas = [ADDTask.gen_data(config[0], config[1], 0.25) for config in test_configs]

        max_length = train_config[1][1]
        for config in val_configs + test_configs:
            if (length:=config[1][1]) > max_length: max_length = length
            
        def process(elements, vocab):
            s = [vocab.bos_token]
            for i, element in enumerate(elements):
                s += [vocab.dash_token]*(max_length - len(str(element))) + [e for e in str(element)]
                
                if i != len(elements)-1:
                    s+= [vocab.sep_token]
            
            s += [vocab.eos_token]
            return np.array(vocab.word_to_index(s), dtype=np.int32)

        return cls(
            'ADD_NEG',
            train_split=Split([
                    Pair(i, process(o, inp_vocab), process(u, label_vocab))
                    for i, (o, u) in enumerate(train_raw_data)
                ],
                Split.SplitType.TRAIN, f'{train_config[1][0]}-{train_config[1][1]}'
            ),
            val_splits=[],
            test_splits=[
                Split([
                    Pair(i, process(o, inp_vocab), process(u, label_vocab))
                    for i, (o, u) in enumerate(raw_data)
                ],
                Split.SplitType.TRAIN, f'{config[1][0]}-{config[1][1]}'
                )
                for (raw_data, config) in zip(test_raw_datas, test_configs)
            ]
        )

        # return cls(
        #     'ADD_NEG',
        #     train_split=NLPSplit.from_data_with_vocab(train_raw_data, Split.SplitType.TRAIN,
        #         f'{train_config[1][0]}-{train_config[1][1]}',
        #         max_length, inp_vocab, label_vocab
        #     ),
        #     val_splits=[],
        #     test_splits=[
        #         NLPSplit.from_data_with_vocab(raw_data, Split.SplitType.TEST, 
        #             f'{config[1][0]}-{config[1][1]}',
        #             max_length, inp_vocab, label_vocab
        #         )
        #         for (raw_data, config) in zip(test_raw_datas, test_configs)
        #     ],
        # )

    # @classmethod
    # def from_config(cls, 
    #     train_config: Tuple[int, Tuple[int, int]],
    #     val_config: List[Tuple[int, Tuple[int, int]]],
    #     test_config: List[Tuple[int, Tuple[int, int]]],
    # ):
    #     max_length = train_config[1][1]
    #     for config in val_config + test_config:
    #         if (l:=config[1][1]) > max_length: max_length = l
    #     max_length += 2  # overflow and negative sign
    #     print('max_length', max_length)

    #     train_split = NLPSplit.from_config(
    #         train_config[0], train_config[1], max_length,
    #         Split.SplitType.TRAIN, f'{train_config[1][0]}-{train_config[1][1]}'
    #     )
    #     inp_vocab, label_vocab = train_split.inp_vocab, train_split.label_vocab

    #     test_splits=[
    #         NLPSplit.from_config(
    #             size, lengths, max_length,
    #             Split.SplitType.TEST, f'{lengths[0]}-{lengths[1]}',
    #             inp_vocab, label_vocab
    #         ) for (size, lengths) in test_config
    #     ]

    #     return cls(
    #         name='ADD',
    #         train_split=train_split,
    #         val_splits=[],
    #         test_splits=test_splits,
    #         # test_splits=[],
    #     )
  

@dataclass
class NLPResult(Result):
    acc: np.ndarray

    def to_disk(self) -> Tuple: return (self.idx, self.out, self.acc)

    @classmethod
    def from_train(cls, idx, pred, acc, vocab: Vocabulary):
        return cls(
            idx,
            # vocab.index_to_word(pred),
            np.array([word.encode('utf-8') for word in vocab.index_to_word(pred)]),
            acc
        )


@dataclass
class NLPSimulation(Simulation):
    
    def __post_init__(self):
        super().__post_init__()

        self.enc_pad_idx = self.task.train_split.inp_vocab.pad_idx
        self.dec_pad_idx = self.task.train_split.label_vocab.pad_idx
        
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.dec_pad_idx
        ).cuda()#.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters())

        self.result_cls = NLPResult

    def _collate_fn(self, batch):
        inp = pad_sequence([torch.LongTensor(b[1]) for b in batch], batch_first=True, padding_value=self.enc_pad_idx)
        label = pad_sequence([torch.LongTensor(b[2]) for b in batch], batch_first=True, padding_value=self.dec_pad_idx)
        
        return {
            'idx': [b[0] for b in batch],
            'inp': inp,
            'label': label
        }

    @classmethod
    def from_task_network_cls(cls,
        task: ADDTask,
        network_cls, network_name
    ):
        return cls(
            network_cls(
                network_name,
                enc_vocab_len=len(task.train_split.inp_vocab),
                enc_pad_idx=task.train_split.inp_vocab.pad_idx,
                dec_vocab_len=len(task.train_split.label_vocab),
                dec_pad_idx=task.train_split.label_vocab.pad_idx,
                is_share_emb=True,
                # mha-related
                d_model=128,
                enc_head=8, enc_layers=6,
                dec_head=8, dec_chead=8, dec_layers=6,
                #
                is_post_norm=False
            ),
            task,
            n_train_steps=1000, val_per_steps=1000, test_per_steps=1000,
            batch_size=256
        )

# if __name__ == '__main__':

# %%
