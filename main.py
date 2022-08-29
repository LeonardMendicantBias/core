# %%

import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from network_library import Net
from trainer_library import Trainer


if __name__ == '__main__':

    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     import torch.distributed as dist
    #     dist.init_process_group(backend="nccl")

    #     local_rank = int(os.environ["LOCAL_RANK"])
    #     local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    # else:
    #     local_rank, local_world_size = 0, 1

    # print('local_rank', local_rank)
    # print('local_world_size', local_world_size)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                         shuffle=True, num_workers=2)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                         shuffle=False, num_workers=2)


    trainer = Trainer()
    trainer.start_training(
        Net(),
        train_set,
        nn.CrossEntropyLoss(reduction='none'),
        optim.Adam#(net.parameters())
    )

    # net = torch.nn.parallel.DistributedDataParallel(
    #     Net().to(local_rank),
    #     device_ids=[local_rank],
    #     output_device=local_rank
    # )
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters())
    
    # for epoch in range(20):  # loop over the dataset multiple times

    #     losses = []
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data

    #         # zero the parameter gradients
    #         optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = net(inputs.to(local_rank))
    #         loss = criterion(outputs, labels.to(local_rank))
    #         loss.backward()
    #         optimizer.step()

    #         losses.append(loss.item())
            
    #         if local_rank == 0:
    #             print(f'\r[{epoch + 1}, {i + 1:5d}] loss: {np.mean(losses):.3f}', end='')
            
    #     if local_rank == 0:
    #         print()

    # print('Finished Training')

# %%
