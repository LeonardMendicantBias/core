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

from .base import Pair, Split, Result, Task, Simulation


@dataclass
class Vocabulary:
    bos_token: str
    eos_token: str
    sep_token: str
    pad_token: str
    unk_token: str
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
            self.pad_token, self.unk_token
        ]
        
        if bool(token_counter):
            for i, token in enumerate(self._special_tokens):
                self.w2i[token] = i
                self.i2w[str(i)] = token
            for name in dict(sorted(token_counter.items())):  # alphabetic sort
                self.w2i[name] = len(self.w2i)
                self.i2w[str(len(self.w2i))] = name
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

    def word_to_index(self, words):
        return [self.w2i.get(w, self.unk_idx) for w in words]

    def index_to_word(self, idx):
        return [self.i2w.get(i, self.unk_token) for i in idx]


@dataclass
class NLPPair(Pair):
    
    def to_arda(self):
        return super().to_arda()


@dataclass
class NLPSplit(Split):
    inp_vocab: Vocabulary
    label_vocab: Vocabulary

    def _gen_add_neg(size: int, lengths: Tuple[int, int], neg_prob: float=0.):
        a = np.random.randint(10**lengths[0], 10**lengths[1], size=size)
        is_neg = np.random.choice([1, -1], size, p=[1.-neg_prob, neg_prob])

        return a * is_neg

    def gen_data(size: int, lengths: Tuple[int, int], neg_prob: float=0.):
        a = NLPSplit._gen_add_neg(size, lengths, neg_prob)
        b = NLPSplit._gen_add_neg(size, lengths, neg_prob)
        
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
                for word in NLPSplit.tokenize(sentence):
                    counter.update(word)
        return counter
    
    def to_disk(self, h5_file: h5py.File) -> None:
        super().to_disk(h5_file)
        if self.split == Split.SplitType.TRAIN:
            # TODO: write vocabularies to disk
            pass

    @classmethod
    def from_config(cls,
        size: int, lengths: Tuple[int, int],
        split: Split.SplitType, name: str,
        inp_vocab=None, label_vocab=None,
        is_share_emb=True
    ):  
        inp, label = NLPSplit.gen_data(size, lengths, 0.25)
        
        if (inp_vocab is None and label_vocab is None) and split == Split.SplitType.TRAIN:
            kw_dict = {
                'bos_token': '[BOS]', 'eos_token': '[EOS]',
                'sep_token': '[SEP]',
                'pad_token': '[PAD]', 'unk_token': '[UNK]',
            }

            inp_token_count = NLPSplit._count(inp)
            label_token_count = NLPSplit._count(label)
            
            if is_share_emb:
                inp_vocab = label_vocab = Vocabulary(**kw_dict, token_counter=inp_token_count + label_token_count)
            else:
                inp_vocab = Vocabulary(**kw_dict, token_counter=inp_token_count)
                label_vocab = Vocabulary(**kw_dict, token_counter=label_token_count)

        def process(elements, vocab):
            s = [vocab.bos_token]
            for i, element in enumerate(elements):
                s += [e for e in str(element)]
                if i != len(elements)-1:
                    s+= [vocab.sep_token]
            s += [vocab.eos_token]
            return np.array(vocab.word_to_index(s), dtype=np.int32)

        assert inp_vocab is not None and label_vocab is not None, "Vocabularies should not be None"
        data = [
            Pair(i, process(o, inp_vocab), process(u, label_vocab))
            for i, (o, u) in enumerate(zip(inp, label))
        ]

        return cls(
            data,
            split=split,
            name=name,
            inp_vocab=inp_vocab,
            label_vocab=label_vocab,
        )
        

@dataclass
class NLPResult(Result):
    acc: np.ndarray

    def to_disk(self) -> Tuple: return (self.idx, self.out, self.acc)

@dataclass
class ADDTask(Task):

    def __post_init__(self):
        super().__post_init__()
    
    @classmethod
    def from_config(cls,
        train_size, train_lengths,
        test_sizes, test_lengthss, 
    ):
        train_split = NLPSplit.from_config(
            train_size, train_lengths,
            Split.SplitType.TRAIN, f'{train_lengths[0]}-{train_lengths[1]}'
        )
        inp_vocab, label_vocab = train_split.inp_vocab, train_split.label_vocab

        test_splits=[
            NLPSplit.from_config(
                size, lengths,
                Split.SplitType.TEST, f'{lengths[0]}-{lengths[1]}',
                inp_vocab, label_vocab
            ) for size, lengths in zip(test_sizes, test_lengthss)
        ]

        return cls(
            name='ADD',
            train_split=train_split,
            val_splits=[],
            test_splits=test_splits,
        )
        

@dataclass
class NLPSimulation(Simulation):
    
    def __post_init__(self):
        super().__post_init__()

        self.enc_pad_idx = self.task.train_split.inp_vocab.pad_idx
        self.dec_pad_idx = self.task.train_split.label_vocab.pad_idx

        self.criterion = nn.CrossEntropyLoss(
            reduction='none',
            ignore_index=self.dec_pad_idx
        ).cuda()#.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters())#.to(self.device)

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
            ),
            task, 
            n_train_steps=1000, val_per_steps=1000, test_per_steps=1000,
            batch_size=8
        )

# if __name__ == '__main__':

# %%
