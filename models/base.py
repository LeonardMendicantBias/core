import os
from functools import cached_property
import h5py
from enum import Enum, unique
from dataclasses import InitVar, dataclass, field
from typing import ClassVar, List, Tuple, Type
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, default_collate
import torch.cuda.amp as amp


class H5Dataset(Dataset):
    '''
        A PyTorch Dataset that reads HDF5
            - support for multiple workers (num_worker > 0)
    '''

    def __init__(self, file_path, split, name):
        self.file_path = file_path
        self.split = split
        self.name = name

        with h5py.File(file_path, "r") as f:
            self.size = f[split][name].attrs['size']

    @cached_property
    def dataset(self): return h5py.File(self.file_path, 'r')[self.split][self.name]

    def __len__(self): return self.size

    def __getitem__(self, idx): return self.dataset['data'][idx]


@dataclass
class Pair:

    idx: np.dtype('int32')
    inp: np.ndarray
    label: np.ndarray

    # TODO: validate type of each sample in __post_init__

    def to_disk(self) -> Tuple: return (self.idx, self.inp, self.label)

    def to_tuple(self) -> Tuple: return (self.idx, self.inp, self.label)


@dataclass
class Split:

    @unique
    class SplitType(Enum):
        TRAIN=1
        VAL=2
        TEST=3
    
    data: List[Type[Pair]]
    split: SplitType
    name: str

    def __len__(self) -> int: return len(self.data)

    def ds(self, file_path) -> Dataset:
        return H5Dataset(
            file_path=file_path,
            split=self.split.name,
            name=self.name,
        )

    def to_disk(self, h5_file: h5py.File) -> None:
        split = h5_file.require_group(self.split.name)
        group = split.require_group(self.name)

        group.attrs['name'] = self.name
        group.attrs['split'] = self.split.name
        group.attrs['size'] = len(self)
        
        dt = np.dtype([
            ('index', np.dtype('int32')),
            ('inp', h5py.vlen_dtype(np.dtype('int32'))),
            ('label', h5py.vlen_dtype(np.dtype('int32'))),
        ])
        ds = group.create_dataset('data', (len(self.data),), dtype=dt)
        for i, sample in enumerate(self.data):
            ds[i] = sample.to_disk()
        
        return group


@dataclass
class Task:
    name: str

    train_split: Type[Split]
    val_splits: List[Type[Split]]=field(default_factory=list)
    test_splits: List[Type[Split]]=field(default_factory=list)

    train_ds: Type[Dataset]=field(init=False)
    val_dss: List[Type[Dataset]]=field(init=False, default_factory=list)
    test_dss: List[Type[Dataset]]=field(init=False, default_factory=list)

    def __post_init__(self):
        self.file_path = f'/home/leonard/deep_learning/core/data/dataset/{self.name}.h5'
        self.ret_path = f'/home/leonard/deep_learning/core/data/result/{self.name}.h5'
        
        self.to_disk()

        self.train_ds = self.train_split.ds(self.file_path)
        self.val_dss = [split.ds(self.file_path) for split in self.val_splits]
        self.test_dss = [split.ds(self.file_path) for split in self.test_splits]

    def to_disk(self) -> None:
        with h5py.File(self.file_path, 'w') as f:
            if os.path.exists(self.ret_path):
                os.remove(self.ret_path)

            self.train_split.to_disk(f)
            [split.to_disk(f) for split in self.val_splits]
            [split.to_disk(f) for split in self.test_splits]

    # def test_to_disk(self, result_split: ResultSplit):
    #     with h5py.File(self.ret_path, 'a') as f:
    #         result_split.to_disk(f)


@dataclass
class Result:
    idx: np.dtype('int32')
    out: np.ndarray

    def to_disk(self) -> Tuple: return (self.idx, self.out)


@dataclass
class ResultSplit:  # ResultList?
    step_idx: int
    network_name: str
    split: Type[Split]
    outputs: List[Type[Result]]=field(init=False, default_factory=list)

    def __len__(self): return len(self.outputs)

    def add_item(self, item: Type[Result]):
        self.outputs.append(item)

    def add_batch(self, items: List[Type[Result]]):
        for item in items:
            self.add_item(item)

    def write_output(self, outputs, net: h5py.Group, step_idx: int):
        dt = np.dtype([
            # ('index', np.dtype('int32')),
            # ('out', h5py.vlen_dtype(np.dtype('int32'))),

            ('index', np.dtype('int32')),
            ('pred', h5py.vlen_dtype(h5py.string_dtype(encoding='utf-8'))),
            ('acc', h5py.vlen_dtype(np.dtype('int32'))),
        ])
        ds = net.create_dataset(str(step_idx), (len(self.outputs),), dtype=dt)
        
        acc = []
        for i, sample in enumerate(outputs):
            s = sample.to_disk()
            ds[i] = s
            acc.append(np.prod(s[-1]))
            
        if (avg:=np.mean(acc)) >= net.attrs.get('acc', -1):
            net.attrs['acc'] = avg
            net.attrs['best'] = ds.ref
        
    def to_disk(self, h5_file: h5py.File):
        # with h5py.File(f'/home/leonard/deep_learning/core/data/result/ADD.h5', 'w') as h5_file:
            # print(self.split)
        split = h5_file.require_group(self.split.split.name)  # TRAIN
        ds = split.require_group(self.split.name)  # 4-4
        # TODO: 'ds' should reference the dataset 
        net = ds.require_group(self.network_name)  # 'Transformer'
        # TODO: 'net' group should save reference to the best performance dataset
        self.write_output(self.outputs, net, self.step_idx)


@dataclass
class Simulation:
    network: torch.nn.Module
    task: Type[Task]

    n_train_steps: int
    val_per_steps: int
    test_per_steps: int

    batch_size: int=256
    num_workers: int=4
    # drop_last: bool=False
    iters_to_accumulate: int=1

    criterion: nn=field(init=False)
    optimizer: torch.optim=field(init=False)

    def __post_init__(self):
        # self.device = 'gpu:0'
        self._is_log = True
        self.file_path = self.task.file_path

        self.network = self._init_network(self.network)

        self.train_loader = self._init_loader(self.task.train_ds)
        self.val_loaders = [self._init_loader(ds) for ds in self.task.val_dss]
        self.test_loaders = [self._init_loader(ds) for ds in self.task.test_dss]

        self.result_cls = Result

    def _init_network(self, network: nn.Module): return network.cuda()#.to(self.device)

    def _init_loader(self, dataset: Dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            # num_workers=self.num_workers,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    def _collate_fn(self, batch): return default_collate(batch)

    def start(self):
        steps = 0
        scaler = amp.GradScaler()
        self.network.train()
        max_epoch = 2
        for epoch in range(max_epoch):
            losses = []
            acc = []
            for i, batch in enumerate(self.train_loader):
                steps += 1
                # inputs, labels = batch.get('inp').to(self.device), batch.get('out').to(self.device)
                inputs, labels = batch.get('inp').cuda(), batch.get('label').cuda()
                
                # forward pass
                with amp.autocast():
                    logits, inp_scores, out_scores, out_cscores = self.network(inputs, labels[:, :-1])
                    loss = self.criterion(
                        logits.view(-1, logits.shape[-1]),
                        labels[:, 1:].contiguous().view(-1)
                    )
                    # loss_ = loss.mean(0) / self.iters_to_accumulate

                    pred = logits.argmax(-1)
                    acc_per_token = pred.eq(labels[:, 1:])
                    a = acc_per_token.prod(dim=-1)
                acc.extend(a.cpu().int().tolist())

                scaler.scale(loss).backward()
                # batch accumulation
                if (i + 1) % self.iters_to_accumulate == 0 or (i + 1 == len(self.train_loader)):
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                losses.append((loss*self.iters_to_accumulate).item())

                if self._is_log:    
                    print(f'\r[{epoch + 1}, {max_epoch}] [{i+1}|{len(self.train_loader)}] loss: {np.mean(losses):.3f} acc: {np.mean(acc):.3f}', end='')
            #     break
            # break
            self.network.eval()
            for j, test_loader in enumerate(self.test_loaders):
                # result_split = ResultSplit(
                #     steps, self.network.name, self.task.test_splits[j]
                # )
                for i, batch in enumerate(test_loader):
                    inputs, labels = batch.get('inp').cuda(), batch.get('label').cuda()
                    with amp.autocast(), torch.no_grad():
                        logits, inp_scores, out_scores, out_cscores = self.network(inputs, labels[:, :-1])
                        pred = logits.argmax(-1)
                        acc_per_token = pred.eq(labels[:, 1:])
                    
                #     results = [
                #         self.result_cls.from_train(idx, p, a, self.task.train_split.label_vocab)
                #         for idx, p, a in zip(
                #             batch.get('idx'),
                #             pred.cpu().int().numpy(),
                #             acc_per_token.cpu().int().numpy()
                #         )
                #     ]
                #     result_split.add_batch(results)

                # self.task.test_to_disk(result_split)
                # break
                    
            self.network.train()
                
            if self._is_log:
                print()


