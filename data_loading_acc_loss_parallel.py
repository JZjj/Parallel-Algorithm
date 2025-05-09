import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool, Queue, Process, freeze_support
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
train_ds = Subset(full_train_ds, list(range(50000)))

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
    process_device = torch.device(f"cuda:{device_id % torch.cuda.device_count()}"
                                  if torch.cuda.is_available() else "cpu")
    
    model = TinyCNN().to(process_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    while True:
        item = data_queue.get()
        if item is None:
            break
            
        images, labels = item
        images, labels = images.to(process_device), labels.to(process_device)
        
        model.train()
        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        
        with torch.no_grad():
            predicted_labels = preds.argmax(dim=1)
            correct = (predicted_labels == labels).sum().item()
            accuracy = correct / labels.size(0)
        
        result_queue.put({'loss': loss.item(), 'accuracy': accuracy, 'samples': labels.size(0)})

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
    
    start_time = time.time()
    
    all_batches = list(loader)

    for batch_data in all_batches:
        data_queue.put(batch_data)
    
    for _ in range(num_workers):
        data_queue.put(None)
 
    batch_results = []
    for _ in range(len(all_batches)):
        batch_results.append(result_queue.get())

    for p in processes:
        p.join()
    
    elapsed_time = time.time() - start_time
    

    total_loss = 0
    total_correct_predictions = 0
    total_samples = 0
    
    for res in batch_results:
        total_loss += res['loss'] * res['samples']
        total_correct_predictions += res['accuracy'] * res['samples']
        total_samples += res['samples']
        
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_accuracy = total_correct_predictions / total_samples if total_samples > 0 else 0
        
    return elapsed_time, avg_loss, avg_accuracy

if __name__ == '__main__':
    freeze_support()

    worker_settings = [1, 2, 4, 6, 8, 12, 16]
    #worker_settings = [1, 2, 4, 6, 8, 12, 16 ,20 , 24, 28, 32]
    
    cpu_cores = multiprocessing.cpu_count()

    times = []
    avg_losses_list = []
    avg_accuracies_list = []

    print(f"Running on device (main process, DataLoader uses this if not overridden in worker): {device}")
    print(f"Number of CPU cores: {cpu_cores}")
    print(f"Total batches to process: {len(list(DataLoader(train_ds, batch_size=batch_size)))}")
    print("-" * 50)

    for workers in worker_settings:
        if workers > cpu_cores:
            print(f"Warning: Using {workers} workers, but only {cpu_cores} CPU cores available. This might lead to suboptimal performance due to context switching.")
        
        print(f"Testing with {workers} worker(s)...")
        elapsed, avg_loss, avg_accuracy = train_with_multiprocessing(workers)
        times.append(elapsed)
        avg_losses_list.append(avg_loss)
        avg_accuracies_list.append(avg_accuracy)
        
        print(f"  Num Workers = {workers}:")
        print(f"    Time Taken  = {elapsed:.2f}s")
        print(f"    Avg Loss    = {avg_loss:.4f}")
        print(f"    Avg Accuracy= {avg_accuracy*100:.2f}%  <---change")
        print("-" * 30)

    if times:
        speedups = [times[0]/t if t > 0 else 0 for t in times]

        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.plot(worker_settings, times, marker='o', label='Training Time')
        plt.xlabel("Number of Workers")
        plt.ylabel("Training Time (s)")
        plt.title("Effect of Worker Count on Training Time")
        plt.xticks(worker_settings)
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(worker_settings, avg_accuracies_list, marker='s', color='green', label='Average Accuracy')
        plt.xlabel("Number of Workers")
        plt.ylabel("Average Accuracy")
        plt.title("Effect of Worker Count on Average Accuracy")
        plt.xticks(worker_settings)
        plt.grid(True)
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(worker_settings, speedups, marker='x', color='red', label='Speedup')
        plt.plot(worker_settings, [w for w in worker_settings], linestyle='--', color='grey', label='Ideal Linear Speedup')
        plt.axhline(y=cpu_cores, color='blue', linestyle=':', label=f'CPU Cores ({cpu_cores})')
        plt.xlabel("Number of Workers")
        plt.ylabel("Speedup (relative to 1 worker)")
        plt.title("Parallelism Speedup")
        plt.xticks(worker_settings)
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig("multiprocessing_performance_accuracy.png")
        plt.show()
    else:
        print("No timing data collected to generate plots.")