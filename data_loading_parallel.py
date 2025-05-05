import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool, Queue, Process
import numpy as np
import os

batch_size = 128
lr = 1e-3
epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

full_train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
train_ds = Subset(full_train_ds, list(range(5000)))

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*14*14, 10),
        )
    def forward(self, x):
        return self.fc(self.conv(x))

def process_batch(batch_idx, data_queue, result_queue, device_id):
    device = torch.device(f"cuda:{device_id % torch.cuda.device_count()}" 
                          if torch.cuda.is_available() else "cpu")
    
    model = TinyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    while True:
        item = data_queue.get()
        if item is None:
            break
            
        images, labels = item
        images, labels = images.to(device), labels.to(device)
        
        model.train()
        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        result_queue.put((batch_idx, loss.item()))

def train_with_multiprocessing(num_workers):
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    data_queue = Queue()
    result_queue = Queue()
    
    processes = []
    for i in range(num_workers):
        p = Process(target=process_batch, args=(i, data_queue, result_queue, i))
        p.start()
        processes.append(p)
    
    start = time.time()
    
    all_batches = list(loader)

    for batch in all_batches:
        data_queue.put(batch)
    
    for _ in range(num_workers):
        data_queue.put(None)
 
    results = []
    for _ in range(len(all_batches)):
        results.append(result_queue.get())

    for p in processes:
        p.join()
    
    elapsed = time.time() - start
    return elapsed

worker_settings = [1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32]
times = []

print(f"Running on device: {device}")
print(f"Number of CPU cores: {multiprocessing.cpu_count()}")

for workers in worker_settings:
    if workers > multiprocessing.cpu_count():
        print(f"Warning: Using {workers} workers but only {multiprocessing.cpu_count()} CPU cores available")
    
    print(f"Testing with {workers} workers...")
    elapsed = train_with_multiprocessing(workers)
    times.append(elapsed)
    print(f"num_workers={workers}: {elapsed:.2f}s")

speedups = [times[0]/time for time in times]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(worker_settings, times, marker='o')
plt.xlabel("Number of Workers")
plt.ylabel("Training Time (s)")
plt.title("Effect of Worker Count on Training Time")
plt.xticks(worker_settings)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(worker_settings, speedups, marker='o', color='green')
plt.xlabel("Number of Workers")
plt.ylabel("Speedup (relative to 1 worker)")
plt.title("Parallelism Speedup")
plt.xticks(worker_settings)
plt.grid(True)

plt.tight_layout()
plt.savefig("multiprocessing_speedup.png")
plt.show()