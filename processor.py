# %%

# import h5py
import os
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

import torch
from torch import nn
import torch.cuda.amp as amp
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

import network_library


@dataclass
class DataList:
    inp: List[np.ndarray]
    out: List[np.ndarray]
    file_path: str
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
            file_path=group.file.filename,
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
            self.sep_token,
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

    def word_to_index(self, words):
        return [self.w2i.get(word, self.w2i.get(self.unk_token)) for word in words]

    def index_to_word(self, idx):
        return [self.i2w.get(index, self.unk_token) for index in idx]


@dataclass
class ProcessDataList:
    inp: List[str]
    out: List[str]
    file_path: str
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
                    '[BOS]', '[EOS]', '[SEP]', '[PAD]', '[UNK]',
                    dict(sorted(counter_dict[key].items()))
                )
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
        process_group = parent_group.require_group('process')
        split = process_group.require_group(self.split)
        group = split.require_group(self.name)

        # group.attrs['file_path'] = parent_group.attrs['file_path']
        group.attrs['name'] = self.name
        group.attrs['split'] = self.split
        group.attrs['size'] = len(self)
        group.attrs['lengths'] = np.array(self.lengths)

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
        
    @classmethod
    def from_datalist(cls, data_list: DataList, is_share_emb=False, train_vocab_dict=None):
        return cls(
            inp=data_list.inp,
            out=data_list.out,
            file_path=data_list.file_path,
            split=data_list.split,
            name=data_list.name,
            lengths=data_list.lengths,
            is_share_emb=is_share_emb,
            train_vocab_dict=train_vocab_dict
        )


class H5Dataset(Dataset):

    def __init__(self, file_path, split, name):
        self.file_path = file_path
        self.dataset = None
        self.split = split
        self.name = name

        with h5py.File(file_path, "r") as f:
            self.size = f['process'][split][name].attrs['size']

    def __del__(self):
        if self.dataset is not None:
            self.dataset.close()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = self.dataset = h5py.File(self.file_path, 'r')#["dataset"]
        data = self.dataset['process'][self.split][self.name]

        return {name: data[name][idx] for name in data}

    @classmethod
    def from_process_data(cls, process_data: ProcessDataList):
        file_path = process_data.file_path
        split = process_data.split
        name = process_data.name
        # vocab_dict = process_data.vocab_dict
        return cls(file_path, split, name)


# class H5DataLoader(DataLoader):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     @classmethod
#     def from_h5dataset(cls, ds: H5Dataset):

#         return cls(

#         )

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
        self.train_loader = self._init_loader(train_set)
        self.test_loaders = [self._init_loader(test_set) for test_set in test_sets]
        
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
    def from_process_data(cls,
        network_constructor: network_library.Transformer,
        ds_file_path: str,
        batch_size=256, num_workers=4,
        iters_to_accumulate=1,
    ):

        with h5py.File(ds_file_path, "r+") as f:
            raw = f['raw']
            
            group = raw['train']
            for n, h5obj in group.items():
                data_list = DataList.from_dataset(h5obj)
                train_set = ProcessDataList.from_datalist(data_list, is_share_emb=True)
                train_set.to_dataset(f)
                vocab_dict = train_set.vocab_dict

            group = raw['test']
            test_sets = []
            for n, h5obj in group.items():
                data_list = DataList.from_dataset(h5obj)
                test_set = ProcessDataList.from_datalist(
                    data_list,
                    is_share_emb=True,
                    train_vocab_dict=vocab_dict
                )
                test_set.to_dataset(f)
                test_sets.append(test_set)

        network = network_constructor(
            len(vocab_dict['inp']), vocab_dict['inp'].pad_idx,
            len(vocab_dict['out']), vocab_dict['out'].pad_idx,
            is_share_emb=True,
            d_model=128,
            enc_head=8, enc_layers=6,
            dec_head=8, dec_chead=8, dec_layers=6,
            # transformer-related
            is_post_norm=True,  # according to original Transformer architecture
            is_enc_abs=True, is_dec_abs=True,
        )

        train_dataset = H5Dataset.from_process_data(train_set)
        train_datasets = [H5Dataset.from_process_data(test_set) for test_set in test_sets]

        return cls(
            network, train_dataset, train_datasets,
            0, 0, 10,
            vocab_dict['inp'].pad_idx, vocab_dict['out'].pad_idx,
            batch_size, num_workers, False, iters_to_accumulate
        )


def _gen_add_neg(size: int, length: Tuple[int, int], neg_prob: float=0.):
    a = np.random.randint(10**length[0], 10**length[1], size=size)
    is_neg = np.random.choice([1, -1], size, p=[1-neg_prob, neg_prob])

    return a * is_neg


def gen_data(size: int, length: Tuple[int, int], neg_prob: float=0.):
    a = _gen_add_neg(size, length, neg_prob)
    b = _gen_add_neg(size, length, neg_prob)
    
    inp = np.stack([a, b], axis=-1)
    out = inp.sum(-1, keepdims=True)

    return inp, out


if __name__ == '__main__':
    # RAW data
    ds_size, length, neg_prob = 100, (0, 2), 0.25

    # save raw dataset to h5
    with h5py.File("./data.h5", "w") as fp:
        fp.attrs['name'] = 'add'

        raw_group = fp.create_group('raw')

        train_data = DataList(
            *gen_data(ds_size, length, neg_prob), fp.filename,
            'train', f'{length[0]}-{length[1]}', length
        )
        train_data.to_dataset(raw_group)

        for l in [3, 4]:
            test_data = DataList(
                *gen_data(ds_size, length, neg_prob), fp.filename,
                'test', f'{l-1}-{l}', (l-1, l)
            )
            test_data.to_dataset(raw_group)

    trainer = Trainer.from_process_data(
        network_library.Transformer, "./data.h5",
        batch_size=8
    )
    trainer.start_training(
        nn.CrossEntropyLoss,
        optim.Adam
    )

# %%
