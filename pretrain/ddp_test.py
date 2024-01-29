import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def example(rank, world_size):
    # create default process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    
    model2 = nn.Linear(10, 10).to(rank)
    
    ddp_model2 = DDP(model2, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    optimizer2 = optim.SGD(ddp_model2.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    outputs = ddp_model2(outputs)
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss=loss_fn(outputs, labels)
    loss.backward()
    # update parameters
    optimizer.step()
    print(loss)

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "0.0.0.0"
    os.environ["MASTER_PORT"] = "7731"
    main()