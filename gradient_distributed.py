#!/usr/bin/env python3
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import spawn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)

def run_worker(rank, world_size):
    dist.init_process_group(
        backend='gloo',
        init_method='tcp://127.0.0.1:29500',
        rank=rank,
        world_size=world_size
    )

    model     = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # dummy data
    x = torch.randn(32, 100)
    y = torch.randint(0, 10, (32,))

    # time only on rank 0
    if rank == 0:
        t0 = time.time()

    # forward
    logits = model(x)
    loss   = criterion(logits, y)

    # compute batch accuracy
    with torch.no_grad():
        preds   = logits.argmax(dim=1)
        correct = preds.eq(y).sum().item()
        acc     = correct / y.size(0) * 100.0

    # backward
    optimizer.zero_grad()
    loss.backward()

    # manual AllReduce: sum then average
    for p in model.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)

    optimizer.step()
    dist.barrier()

    if rank == 0:
        elapsed = time.time() - t0
        print(f"[world_size={world_size}] step time: {elapsed:.4f} s, batch accuracy: {acc:.2f}%")

    dist.destroy_process_group()

if __name__ == "__main__":
    for world_size in (1, 2, 4, 6, 8, 10, 12, 14, 16):
        print(f"\n>>> Benchmarking manual AllReduce with world_size={world_size}")
        spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
