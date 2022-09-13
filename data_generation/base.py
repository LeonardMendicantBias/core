# %%
import h5py
from enum import Enum, unique
from dataclasses import dataclass, InitVar, field
from typing import List, Tuple, Dict
import numpy as np
from collections import Counter, OrderedDict


@unique
class SplitType(Enum):
    TRAIN=1
    VAL=2
    TEST=3


@unique
class Kind(Enum):
    INP=1
    OUT=2


@dataclass
class DataSplit:
    '''
        Define a data split either (train, val, or test)
    '''
    inp: List[np.ndarray]
    out: List[np.ndarray]
    split: SplitType
    name: str

    def __post_init__(self):
        assert (i:=len(self.inp)) == (j:=len(self.out)), f"sizes should be equal but {i} and {j}"

    def __len__(self):
        return len(self.inp)
        
    # write instance attributes under 'split' group under 'name' group
    def to_dataset(self, h5_file: h5py.File):
        raw_group = h5_file.require_group('raw')
        split = raw_group.require_group(self.split.name)
        group = split.require_group(self.name)

        group.attrs['split'] = self.split.value
        group.attrs['name'] = self.name

        group.create_dataset('inp', data=self.inp)
        group.create_dataset('out', data=self.out)

    # receive a 'name' group and parse as a DataList
    @classmethod
    def from_dataset(cls, h5_file: h5py.File, split: SplitType):
        raw_group = h5_file.require_group('raw')
        split = raw_group.require_group(split.name)
        l = []
        for name in split:
            group = split.require_group(name)
            l.append(
                cls(
                    inp=group['inp'],
                    out=group['out'],
                    split=SplitType(group.attrs['split']),
                    name=group.attrs['name'],
                )
            )
        if len(l) == 1:
            return l[0]
        return l


@dataclass
class Vocabulary:
    bos_token: str=None
    eos_token: str=None
    sep_token: str=None
    pad_token: str=None
    unk_token: str=None

    name: str=None

    word_counter: InitVar[Counter] = None

    w2i: Dict = field(repr=False, default_factory=dict)
    i2w: Dict = field(repr=False, default_factory=dict)

    def __post_init__(self, word_counter):
        self._special_tokens = [
            self.bos_token, self.eos_token,
            self.sep_token,
            self.pad_token, self.unk_token
        ]

        self.w2i = {token: i for i, token in enumerate(self._special_tokens)}

        if word_counter is not None:
            for key, _ in word_counter.items():
                if key not in self.w2i:
                    self.w2i[key] = len(self.w2i)

        self.i2w = {self.w2i[k]:k for k in self.w2i}

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

    def to_dataset(self, h5_file: h5py.File):
        process_group = h5_file.require_group('vocab')
        group = process_group.create_group(self.name)
        group.create_dataset(
            'words',
            data=list(self.w2i.keys())
        )
        group.create_dataset(
            'indices',
            data=list(self.w2i.values())
        )

    @classmethod
    def from_dataset(cls, h5_file: h5py.File, name):
        process_group = h5_file.require_group('vocab')
        group = process_group.require_group(name)

        words = group['words']
        indices = group['indices']

        w2i = {w: i for w, i in zip(words, indices)}
        i2w = {w2i[k]:k for k in w2i}
        return cls(
            w2i=dict(sorted(w2i.items())),
            i2w=dict(sorted(i2w.items()))
        )
        

@dataclass
class ProcessDataSplit:
    '''
        process the raw data for NLP task
            - Count words and produce vocabulary
            - Tokenize raw data and map into indices
    '''
    inp: List[List[int]]
    out: List[List[int]]
    split: SplitType
    name: str
    ####
    vocab_dict: Dict[Vocabulary, Vocabulary] = field(
        repr=False, #init=True,
        default_factory=lambda: {},
    )

    # is_share_emb: InitVar[bool]=False
    cache_dir: InitVar[str]=None

    def __post_init__(self, cache_dir):
        assert (i:=len(self.inp)) == (j:=len(self.out)), f"sizes should be equal but {i} and {j}"
        
        if cache_dir is not None:
            with h5py.File(cache_dir) as f:
                self.to_dataset(f)
    
    @staticmethod
    def process(elements, vocab: Vocabulary):
        s = [vocab.bos_token]
        for i, element in enumerate(elements):
            s += [e for e in str(element)]
            if i != len(elements)-1:
                s+= [vocab.sep_token]
        s += [vocab.eos_token]

        return np.array(vocab.word_to_index(s), dtype=np.int32)

    def __len__(self): return len(self.inp)

    @staticmethod
    def _count(elements):
        counter = Counter()
        for element in elements:
            for e in element:
                for s in str(e):
                    counter.update(s)
        return counter

    def to_dataset(self, h5_file: h5py.File):
        process_group = h5_file.require_group('process')
        split = process_group.require_group(self.split.name)
        group = split.require_group(self.name)

        group.attrs['name'] = self.name
        group.attrs['split'] = self.split.value
        group.attrs['size'] = len(self)

        group.create_dataset(
            'inp',
            dtype=h5py.vlen_dtype(np.dtype('int32')),
            data=self.inp
        )
        group.create_dataset(
            'out',
            dtype=h5py.vlen_dtype(np.dtype('int32')),
            data=self.out
        )
        
        if self.split == SplitType.TRAIN:
            for name, vocab in self.vocab_dict.items():
                vocab.to_dataset(h5_file)
            
    @classmethod
    def from_dataset(cls, h5_file: h5py.Group, split: SplitType):
        process_group = h5_file.require_group('process')
        split = process_group.get(split.name, [])
        l = []
        for name in split:
            vocab_dict = {
                'inp': Vocabulary.from_dataset(h5_file, 'inp'),
                'out': Vocabulary.from_dataset(h5_file, 'out'),
            }

            group = split.require_group(name)
            l.append(
                cls(
                    inp=group['inp'],
                    out=group['out'],
                    split=SplitType(group.attrs['split']),
                    name=group.attrs['name'],
                    vocab_dict=vocab_dict
                )
            )
        if len(l) == 1:
            return l[0]
        return l
        
    @classmethod
    def from_datalist(cls, data_list: DataSplit, is_share_emb=False, vocab_dict={}):

        if not bool(vocab_dict) and data_list.split == SplitType.TRAIN:
            counter_dict = {
                'inp': ProcessDataSplit._count(data_list.inp),
                'out': ProcessDataSplit._count(data_list.out)
            }
            
            if is_share_emb:
                share_counter = counter_dict['inp'] + counter_dict['out']
                counter_dict['inp'] = share_counter
                counter_dict['out'] = share_counter
            
            for key, counter in counter_dict.items():
                vocab_dict[key] = Vocabulary(
                    '[BOS]', '[EOS]', '[SEP]', '[PAD]', '[UNK]',
                    key,
                    dict(sorted(counter.items()))
                )

        assert vocab_dict, 'Vocabulary required'
        
        return cls(
            inp=[ProcessDataSplit.process(element, vocab_dict['inp']) for element in data_list.inp],
            out=[ProcessDataSplit.process(element, vocab_dict['out']) for element in data_list.out],
            split=data_list.split,
            name=data_list.name,
            vocab_dict=vocab_dict
        )


if __name__ == '__main__':
    np.random.seed(10**0)

    ds_size, lengths, neg_prob = 100, (0, 2), 0.25
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
        
    
    with h5py.File("/mount/dataset/ADD.h5", "w") as fp:
        train_data = DataSplit(
            *gen_data(ds_size, lengths, neg_prob),
            SplitType.TRAIN, f'{lengths[0]}-{lengths[1]}'
        )
        train_data.to_dataset(fp)
        process_train_data = ProcessDataSplit.from_datalist(
            train_data, is_share_emb=True,
        )
        process_train_data.to_dataset(fp)

        test_data = DataSplit(
            *gen_data(ds_size, (2, 3), neg_prob),
            SplitType.TEST, f'{2}-{3}'
        )
        test_data.to_dataset(fp)
        process_test_data = ProcessDataSplit.from_datalist(
            test_data, True, process_train_data.vocab_dict,
        )
        process_test_data.to_dataset(fp)

# %%

# %%
