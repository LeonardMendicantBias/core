# %%

import os
from random import shuffle
import torch
import torch.cuda.amp as amp

import numpy as np


"""
    Trainer class trains a network with given loaders.
    Trainer aims to maximize speed via by using AMP.
    It also perform logging.
"""
class Trainer:

    def __init__(self):
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            import torch.distributed as dist
            dist.init_process_group(backend="nccl")

        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

        self.iters_to_accumulate = 5

        self._is_log = self.local_rank == 0
        if self._is_log:
            print(f'start training on {self.local_world_size} GPUs')
            
        
    def _init(self, network, dataset):
        is_distributed = self.local_world_size > 1
        
        batch_size = 32
        network = torch.nn.parallel.DistributedDataParallel(
            network.to(self.local_rank),
            device_ids=[self.local_rank],
            output_device=self.local_rank
        ) if is_distributed else network.to(self.local_rank)

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=~is_distributed,
            num_workers=0 if is_distributed else 4,
            sampler=torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.local_world_size,
                rank=self.local_rank
            ) if is_distributed else None
        )

        return network, train_loader
        
    def start_training(self,
        network,
        train_set,
        criterion,
        optimizer_init,
    ):
        network, train_loader = self._init(network, train_set)
        optimizer = optimizer_init(network.parameters())
        optimizer.zero_grad()  # just to make sure

        scaler = amp.GradScaler()
        for epoch in range(20):
            losses = []
            for i, batch in enumerate(train_loader):
                inputs, labels = batch

                with amp.autocast():
                    outputs = network(inputs.to(self.local_rank))
                    loss = criterion(outputs, labels.to(self.local_rank))
                    loss_ = loss.mean(0) / self.iters_to_accumulate

                scaler.scale(loss_).backward()

                if (i + 1) % self.iters_to_accumulate == 0 or (i + 1 == len(train_loader)):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                losses.append((loss_*self.iters_to_accumulate).item())
                
                if self._is_log:
                    print(f'\r[{epoch + 1}, {i + 1:5d}] loss: {np.mean(losses):.3f}', end='')
                
            if self._is_log:
                print()

        if self._is_log:
            print('Finished Training')



