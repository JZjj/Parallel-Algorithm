import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
import matplotlib.pyplot as plt

NUM_EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001

class SimpleModelParallelMLP(nn.Module):
    def __init__(self, num_gpus=1):
        super(SimpleModelParallelMLP, self).__init__()
        self.num_gpus = num_gpus
        hidden_size1 = 512
        hidden_size2 = 512
        hidden_size3 = 512

        if self.num_gpus <= 1:
            self.device0 = torch.device("cuda:0")
            self.device1 = torch.device("cuda:0")
            self.layer1 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28, hidden_size1),
                nn.ReLU()
            ).to(self.device0)
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_size1, hidden_size2),
                nn.ReLU()
            ).to(self.device0)
            self.layer3 = nn.Sequential(
                nn.Linear(hidden_size2, hidden_size3),
                nn.ReLU()
            ).to(self.device0)
            self.output_layer = nn.Linear(hidden_size3, 10).to(self.device0)
        elif self.num_gpus >= 2:
            self.device0 = torch.device("cuda:0")
            self.device1 = torch.device("cuda:1")
            self.layer1 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28, hidden_size1),
                nn.ReLU()
            ).to(self.device0)
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_size1, hidden_size2),
                nn.ReLU()
            ).to(self.device0)
            self.layer3 = nn.Sequential(
                nn.Linear(hidden_size2, hidden_size3),
                nn.ReLU()
            ).to(self.device1)
            self.output_layer = nn.Linear(hidden_size3, 10).to(self.device1)
        else:
            raise ValueError("Number of GPUs must be at least 1.")

    def forward(self, x):
        if self.num_gpus <= 1:
            x = x.to(self.device0)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.output_layer(x)
            return x
        elif self.num_gpus >= 2:
            x = x.to(self.device0)
            x = self.layer1(x)
            x = self.layer2(x)
            x = x.to(self.device1)
            x = self.layer3(x)
            x = self.output_layer(x)
            return x

def get_mnist_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    os.makedirs('./data', exist_ok=True)
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader

def train_model(model, train_loader, optimizer, criterion):
    model.train()
    start_time = time.time()
    epoch_losses = []
    epoch_accs = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            input_device = next(model.parameters()).device
            data = data.to(input_device)
            target = target.to(input_device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_losses.append(epoch_loss)
        epoch_accs.append(epoch_acc)
    end_time = time.time()
    total_duration = end_time - start_time
    return total_duration, epoch_losses, epoch_accs

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("Error: CUDA not detected. This script requires GPU(s).")
        exit()

    num_available_gpus = torch.cuda.device_count()

    if num_available_gpus == 0:
        print("Error: No GPU found.")
        exit()

    train_loader = get_mnist_loaders(BATCH_SIZE)

    model_1gpu = SimpleModelParallelMLP(num_gpus=1)
    optimizer_1gpu = optim.Adam(model_1gpu.parameters(), lr=LEARNING_RATE)
    criterion_1gpu = nn.CrossEntropyLoss().to(next(model_1gpu.parameters()).device)
    time_1gpu, losses_1gpu, accs_1gpu = train_model(model_1gpu, train_loader, optimizer_1gpu, criterion_1gpu)

    time_2gpu = None
    losses_2gpu = []
    accs_2gpu = []
    speedup = None

    if num_available_gpus >= 2:
        model_2gpu = SimpleModelParallelMLP(num_gpus=2)
        optimizer_2gpu = optim.Adam(model_2gpu.parameters(), lr=LEARNING_RATE)
        criterion_2gpu = nn.CrossEntropyLoss().to(model_2gpu.device1)
        time_2gpu, losses_2gpu, accs_2gpu = train_model(model_2gpu, train_loader, optimizer_2gpu, criterion_2gpu)

        if time_2gpu is not None and time_2gpu > 0:
            speedup = time_1gpu / time_2gpu

        print(f"Single GPU Time (T1): {time_1gpu:.4f} seconds")
        print(f"Dual GPU Time (T2): {time_2gpu:.4f} seconds")
        if speedup is not None:
            print(f"Speedup (T1 / T2): {speedup:.2f}x")

        labels = ['1 GPU', '2 GPUs (Model Parallel)']
        times = [time_1gpu, time_2gpu]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
        ax1.bar(labels, times, color=['blue', 'orange'])
        ax1.set_ylabel('Training Time (seconds)')
        ax1.set_title('Training Time Comparison')
        for bar in ax1.containers[0]:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}s', va='bottom', ha='center')

        epochs = range(1, NUM_EPOCHS + 1)
        ax2.plot(epochs, accs_1gpu, 'b-', label='1 GPU')
        ax2.plot(epochs, accs_2gpu, 'r-', label='2 GPUs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training Accuracy Comparison')
        ax2.legend()

        ax3.plot(epochs, losses_1gpu, 'b-', label='1 GPU')
        ax3.plot(epochs, losses_2gpu, 'r-', label='2 GPUs')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss Comparison')
        ax3.legend()

        plt.tight_layout()
        plt.savefig('comparison.png')
    else:
        print(f"Single GPU Time (T1): {time_1gpu:.4f} seconds")