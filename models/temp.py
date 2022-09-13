# %%

import h5py
from enum import Enum, unique
from dataclasses import InitVar, dataclass, field
from typing import List, Tuple, Dict, Union
import numpy as np
from collections import Counter, OrderedDict
from torch.utils.data import Dataset, DataLoader

import os
import torch
from torch import nn
import torch.cuda.amp as amp
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import network_library


@unique
class SplitType(Enum):
    TRAIN=1
    VAL=2
    TEST=3


class H5Dataset(Dataset):

    def __init__(self, file_path, split, name):
        self.file_path = file_path
        self.dataset = None
        self.split = split
        self.name = name

        with h5py.File(file_path, "r") as f:
            self.size = f['process'][split][name].attrs['size']

    def __del__(self):
        if self.dataset is not None: self.dataset.close()

    def __len__(self): return self.size

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = self.dataset = h5py.File(self.file_path, 'r')#["dataset"]
        data = self.dataset['process'][self.split][self.name]

        return {name: data[name][idx] for name in data}


# @dataclass
# class DataList:
#     data: List[np.ndarray]

#     def __len__(self): return len(self.data)


@dataclass
class DataSplit:
    
    inp: List[np.ndarray]
    out: List[np.ndarray]
    split: SplitType
    name: str

    def __post_init__(self):
        assert (i:=len(self.inp)) == (j:=len(self.out)), f"sizes should be equal but {i} and {j}"

    def to_disk(self, h5_file: h5py.File, name: str):
        raw_group = h5_file.require_group('raw')
        split = raw_group.require_group(self.split.name)
        group = split.require_group(name)

        group.attrs['split'] = self.split.value
        group.attrs['name'] = name

        group.create_dataset('inp', data=self.inp)
        group.create_dataset('out', data=self.out)

    @classmethod
    def from_disk(cls, h5_file: h5py.File, split: SplitType, name: str):
        raw_group = h5_file.require_group('raw')
        split = raw_group.require_group(split.name)
        group = split.require_group(name)
        return cls(
            inp=group['inp'],
            out=group['out'],
            split=SplitType(group.attrs['split']),
        )


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


class Singleton (type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SingleVocabulary(Vocabulary, metaclass=Singleton):
    pass


@dataclass
class NLPData:

    # Only NLP data need tokenization
    @dataclass
    class TokenizedData:
        data: List[np.ndarray]  # A set of tokenized sentences
        vocab: Union[Vocabulary, SingleVocabulary]=field(default=None)

        is_share_emb: InitVar[bool]=False

        def __len__(self): return len(self.data)

        @staticmethod
        def tokenize(s): return str(s)

        @staticmethod
        def _count(elements):
            counter = Counter()
            for sample in elements:
                for sentence in sample:
                    for word in NLPData.TokenizedData.tokenize(sentence):
                        counter.update(word)
            return counter

        def __post_init__(self, is_share_emb):
            if self.vocab is None:
                token_counter = NLPData.TokenizedData._count(self.data)
                kw_dict = {
                    'bos_token': '[BOS]', 'eos_token': '[EOS]',
                    'sep_token': '[SEP]',
                    'pad_token': '[PAD]', 'unk_token': '[UNK]',
                    'token_counter': token_counter,
                }
                self.vocab = SingleVocabulary(**kw_dict) if is_share_emb else Vocabulary(**kw_dict)
            
            def process(elements):
                s = [self.vocab.bos_token]
                for i, element in enumerate(elements):
                    s += [e for e in str(element)]
                    if i != len(elements)-1:
                        s+= [self.vocab.sep_token]
                s += [self.vocab.eos_token]
                return np.array(self.vocab.word_to_index(s), dtype=np.int32)

            self.data = [process(sample) for sample in self.data]

    inp: TokenizedData
    out: TokenizedData

    split: SplitType
    name: str
    file_path: str=field(init=False)

    def __post_init__(self):
        assert (i:=len(self.inp)) == (j:=len(self.out)), f"sizes should be equal but {i} and {j}"
        # write to disk to enable HDF5-based PyTorch Dataset

        self.file_path = f"/mount/dataset/ADD.h5"
        self.to_disk()

    def __len__(self): return len(self.inp)

    @property
    def ds(self):
        return H5Dataset(
            file_path=self.file_path,
            split=self.split.name,
            name=self.name,
        )

    def to_disk(self):
        with h5py.File(self.file_path, "a") as f:
            f = f.require_group('process')
            split = f.require_group(self.split.name)
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
        print('process', self.split.name, self.name)
        
        # with h5py.File(self.file_path, "r") as f:
        #     for name in f:
        #         print('-', name)
        #     for name in f['process']:
        #         print('process', name)

    @classmethod
    def from_data_split(cls,
        data: DataSplit,
        inp_vocab: Vocabulary=None, out_vocab: Vocabulary=None,
        is_share_emb=False
    ):
        return cls(
            inp=NLPData.TokenizedData(data.inp, inp_vocab, is_share_emb),
            out=NLPData.TokenizedData(data.out, out_vocab, is_share_emb),
            split=SplitType(data.split),
            name=data.name,
        )

    @classmethod
    def from_numpy(cls,
        inp: List[np.ndarray],
        out: List[np.ndarray],
        split: SplitType,
        name: str,
        is_share_emb=False
    ):
        datasplit = DataSplit(inp, out, split, name)
        return cls.from_data_split(datasplit, is_share_emb)


@dataclass
class Task:

    train_set: NLPData
    val_sets: List[NLPData]
    test_sets: List[NLPData]

    def __post_init__(self):
        pass

    @classmethod
    def from_datasplit(cls,
        train_split: DataSplit,
        val_splits: List[DataSplit],
        test_splits: List[DataSplit],
        is_share_emb=False
    ):
        train_set = NLPData.from_data_split(train_split, is_share_emb=is_share_emb)
        val_sets = [NLPData.from_data_split(split, train_set.inp.vocab, train_set.out.vocab) for split in val_splits]
        test_sets = [NLPData.from_data_split(split, train_set.inp.vocab, train_set.out.vocab) for split in test_splits]

        return cls(
            train_set, val_sets, test_sets
        )


class Trainer:

    def __init__(self,
        network: nn.Module,
        train_set: Dataset, test_sets: List[Dataset],
        n_train_steps, val_per_step, test_per_step,
        enc_pad_idx, dec_pad_idx,
        batch_size=256, num_workers=4,
        drop_last=False,
        iters_to_accumulate=1,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.n_train_steps = n_train_steps
        self.val_per_step = val_per_step
        self.test_per_step = test_per_step
        self.iters_to_accumulate = iters_to_accumulate
        self.pad_idx_dict = {
            'inp': enc_pad_idx,
            'out': dec_pad_idx,
        }

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            import torch.distributed as dist
            dist.init_process_group(backend="nccl")

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.is_distributed = self.local_world_size > 1

        self._is_log = (self.local_rank == 0)
        if self._is_log:
            print(f'start training on {self.local_world_size} GPUs {self.local_rank} {self.local_world_size}')

        self.network = self._init_network(network)
        self.train_loader = self._init_loader(train_set.ds)
        self.test_loaders = [self._init_loader(test_set.ds) for test_set in test_sets]
        
    def _init_network(self, network: nn.Module):
        network = network.to(self.local_rank)
        return torch.nn.parallel.DistributedDataParallel(
            network,
            device_ids=[self.local_rank],
            output_device=self.local_rank
        ) if self.is_distributed else network
        
    def _collate_fn(self, batch):
        inp = pad_sequence([torch.LongTensor(b['inp']) for b in batch], batch_first=True, padding_value=self.pad_idx_dict['inp'])
        out = pad_sequence([torch.LongTensor(b['out']) for b in batch], batch_first=True, padding_value=self.pad_idx_dict['out'])
        
        return {
            'inp': inp,
            'out': out
        }

    def _init_loader(self, dataset: Dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            # shuffle=False if self.is_distributed else True,
            shuffle=not self.is_distributed,
            collate_fn=self._collate_fn,
            # num_workers=0 if self.is_distributed else self.num_workers,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=self.drop_last,
            sampler=torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.local_world_size,
                rank=self.local_rank
            ) if self.is_distributed else None
        )
        
    def start_training(self,
        criterion_init,
        optimizer_init,
    ):
        criterion = criterion_init(reduction='none', ignore_index=self.pad_idx_dict['out'])
        optimizer = optimizer_init(self.network.parameters())
        optimizer.zero_grad()  # just to make sure

        steps = 0
        scaler = amp.GradScaler()
        self.network.train()
        max_epoch = 5
        for epoch in range(max_epoch):
            losses = []
            for i, batch in enumerate(self.train_loader):
                steps += 1
                inputs, labels = batch.get('inp').to(self.local_rank), batch.get('out').to(self.local_rank)

                # forward pass
                with amp.autocast():
                    logits, inp_scores, out_scores, out_cscores = self.network(inputs, labels[:, :-1])
                    loss = criterion(
                        logits.view(-1, logits.shape[-1]),
                        labels[:, 1:].contiguous().view(-1)
                    )
                    loss_ = loss.mean(0) / self.iters_to_accumulate
                scaler.scale(loss_).backward()

                # batch accumulation
                if (i + 1) % self.iters_to_accumulate == 0 or (i + 1 == len(self.train_loader)):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                losses.append((loss_*self.iters_to_accumulate).item())

                if self._is_log:
                    print(f'\r[{epoch + 1}, {max_epoch}] [{i}|{len(self.train_loader)}] loss: {np.mean(losses):.3f}', end='')
                
                # if steps % self.test_per_step == 0:
            self.network.eval()
            for test_loader in self.test_loaders:
                for i, batch in enumerate(test_loader):
                    with amp.autocast(), torch.no_grad():
                        logits, inp_scores, out_scores, out_cscores = self.network(inputs, labels[:, :-1])
                        pred = logits.argmax(-1)
                        acc_per_token = pred.eq(labels[:, :-1])
                        acc = acc_per_token.all(dim=-1)
                    # print(acc)
                    
            self.network.train()
                
            if self._is_log:
                print()
    
    @classmethod
    def from_task(cls, 
        network_constructor: network_library.Transformer,
        task: Task
    ):

        network = network_constructor(
            len(task.train_set.inp.vocab), task.train_set.inp.vocab.pad_idx,
            len(task.train_set.out.vocab), task.train_set.out.vocab.pad_idx,
            is_share_emb=True,
            d_model=128,
            enc_head=8, enc_layers=6,
            dec_head=8, dec_chead=8, dec_layers=6,
            # transformer-related
            is_post_norm=True,  # according to original Transformer architecture
            is_enc_abs=True, is_dec_abs=True,
        )

        return cls(
            network, task.train_set, task.test_sets,
            0, 0, 10,
            task.train_set.inp.vocab.pad_idx, task.train_set.out.vocab.pad_idx,
            16, 4, False, 1
        )


if __name__ == '__main__':

    if os.path.exists('/mount/dataset/ADD.h5'):
        os.remove('/mount/dataset/ADD.h5')

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
        

    train_split = DataSplit(
        *gen_data(ds_size, lengths, neg_prob),
        SplitType.TRAIN, f'{lengths[0]}-{lengths[1]}'
    )
    test_splits = [
        DataSplit(
            *gen_data(20, (2,3), neg_prob),
            SplitType.TEST, f'2-3'
        ),
        DataSplit(
            *gen_data(20, (3,4), neg_prob),
            SplitType.TEST, f'3-4'
        )
    ]
    task = Task.from_datasplit(train_split, [], test_splits)

    trainer = Trainer.from_task(
        network_library.Transformer, task
    )
    trainer.start_training(
        nn.CrossEntropyLoss,
        optim.Adam
    )
        

# %%
