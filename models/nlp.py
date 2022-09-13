# %%
from re import T
import h5py
from dataclasses import field, dataclass, InitVar
from collections import Counter
from typing import List, Union, Tuple, Dict
import numpy as np

from .base import Singleton, Task, Split, Data


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


class SingleVocabulary(Vocabulary, metaclass=Singleton):
    pass

@dataclass
class TokenizedData(Data):
    vocab: Union[Vocabulary, SingleVocabulary]=field(default=None)
    
    @staticmethod
    def tokenize(s): return str(s)

    @staticmethod
    def _count(elements):
        counter = Counter()
        for sample in elements:
            for sentence in sample:
                for word in NLPTask.NLPSplit.TokenizedData.tokenize(sentence):
                    counter.update(word)
        return counter

    @classmethod
    def from_raw(cls,
        data: List[np.ndarray],
        vocab: Union[Vocabulary, SingleVocabulary]=None,
        is_share_emb: bool=False
    ):

        if vocab is None:
            token_counter = NLPTask.NLPSplit.TokenizedData._count(data)
            kw_dict = {
                'bos_token': '[BOS]', 'eos_token': '[EOS]',
                'sep_token': '[SEP]',
                'pad_token': '[PAD]', 'unk_token': '[UNK]',
                'token_counter': token_counter,
            }
            vocab = SingleVocabulary(**kw_dict) if is_share_emb else Vocabulary(**kw_dict)
            
        assert vocab is not None, 'vocab is None'
        
        def process(elements):
            s = [vocab.bos_token]
            for i, element in enumerate(elements):
                s += [e for e in str(element)]
                if i != len(elements)-1:
                    s+= [vocab.sep_token]
            s += [vocab.eos_token]
            return np.array(vocab.word_to_index(s), dtype=np.int32)

        return cls(
            [process(sample) for sample in data],
            vocab
        )

@dataclass
class NLPSplit(Split):

    @classmethod
    def from_config(cls,
        size: int, lengths: Tuple[int, int],
        split: Task.Split.SplitType, name: str,
        inp_vocab=None, out_vocab=None,
        is_share_emb=True
    ):
        
        def _gen_add_neg(size: int, lengths: Tuple[int, int], neg_prob: float=0.):
            a = np.random.randint(10**lengths[0], 10**lengths[1], size=size)
            is_neg = np.random.choice([1, -1], size, p=[1-neg_prob, neg_prob])

            return a * is_neg

        def gen_data(size: int, lengths: Tuple[int, int], neg_prob: float=0.):
            a = _gen_add_neg(size, lengths, neg_prob)
            b = _gen_add_neg(size, lengths, neg_prob)
            
            inp = np.stack([a, b], axis=-1)
            out = inp.sum(-1, keepdims=True)

            return inp, out
        
        inp, out = gen_data(size, lengths, 0.25)
        inp = cls.TokenizedData.from_raw(inp, inp_vocab, is_share_emb)
        out = cls.TokenizedData.from_raw(out, out_vocab, is_share_emb)

        return cls(
            inp, out,
            split=split,
            name=name,
        )


@dataclass
class NLPTask(Task):

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
        inp_vocab, out_vocab = train_split.inp.vocab, train_split.out.vocab

        test_splits=[
            NLPSplit.from_config(
                size, lengths,
                Split.SplitType.TEST, f'{lengths[0]}-{lengths[1]}',
                inp_vocab, out_vocab
            ) for size, lengths in zip(test_sizes, test_lengthss)
        ]

        return cls(
            name='ADD',
            train_split=train_split,
            val_splits=[],
            test_splits=test_splits,
        )
        
