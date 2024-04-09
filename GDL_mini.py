import torch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, ChebConv
import torch.nn.functional as F
import torch.nn as nn



import numpy as np
import random

def fix_seed(seed_value=42):
    random.seed(seed_value)  # Python random module.
    np.random.seed(seed_value)  # Numpy module.
    torch.manual_seed(seed_value)  # PyTorch to initialize the weights.
    
    # if you are using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True  # fixes the GPU to deterministic mode.
        torch.backends.cudnn.benchmark = False

# Call this function at the beginning of your script.
fix_seed(42)



train_dataset = MNISTSuperpixels(root='dataset/MNIST/', train=True)
test_dataset = MNISTSuperpixels(root='dataset/MNIST/', train=False)

class GCNModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.fc1 = torch.nn.Linear(num_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.fc2 = torch.nn.Linear(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Applying the first convolution and skip connection
        skip1 = self.fc1(x)  # Compute skip connection first
        x = self.conv1(x, edge_index) # Apply non-linearity after conv

        x = x + skip1  # Add skip connection result

        x = F.relu(x)

        # Second convolution and skip connection
        skip2 = self.fc2(x)  # Compute skip connection from output of previous layer
        x = self.conv2(x, edge_index)  # Apply non-linearity after conv

        x = x + skip2  # Add skip connection result

        x = F.relu(x)

        # Third convolution and skip connection
        skip3 = self.fc3(x)  # Compute skip connection from output of previous layer
        x = self.conv3(x, edge_index)  # Apply non-linearity after conv

        x = x + skip3  # Add skip connection result

        x = F.relu(x)

        x = global_mean_pool(x, batch)  # Apply pooling to summarize the graph features
        x = self.fc(x)
        return x

class ChebNetModel(torch.nn.Module):
    def __init__(self, num_features, num_classes, K=2):
        super(ChebNetModel, self).__init__()
        self.K = 2
        self.conv1 = ChebConv(num_features, 64, self.K)
        self.conv2 = ChebConv(64, 128, self.K)
        self.conv3 = ChebConv(128, 128 , self.K)
        self.pool = global_mean_pool
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # No need to pass num_nodes to ChebConv layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.pool(x, batch)  # Pooling
        x = self.fc(x)  # Final fully connected layer
        return x

def train(model, train_loader, optimizer, criterion, device, epochs=50):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {epoch_loss}')
    return train_losses

def test(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data).max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print(f'Accuracy: {accuracy}')
    return accuracy


train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

num_features = train_dataset.num_features
num_classes = 10

gcn_model = GCNModel(num_features, num_classes).to('cuda')
chebnet_model = ChebNetModel(num_features, num_classes, K=2).to('cuda')

criterion = torch.nn.CrossEntropyLoss()
gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.001)
chebnet_optimizer = torch.optim.Adam(chebnet_model.parameters(), lr=0.001)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize and train GCN Model
gcn_losses = train(gcn_model, train_loader, gcn_optimizer, criterion, device, epochs=2000)
gcn_accuracy = test(gcn_model, test_loader, device)

# Initialize and train ChebNet Model
chebnet_losses = train(chebnet_model, train_loader, chebnet_optimizer, criterion, device, epochs=2000)
chebnet_accuracy = test(chebnet_model, test_loader, device)

import matplotlib.pyplot as plt

# Ensure the matplotlib backend is set up correctly in headless environments
plt.switch_backend('agg')

# Plot training losses and save the figure
plt.figure(figsize=(12, 6))
plt.plot(gcn_losses, label='GCN with skip connections Loss')
plt.plot(chebnet_losses, label='ChebNet Loss')
plt.title('Training Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss_comparison_with_skip.png')  # Saves the plot to a file
plt.close()  # Close the figure to free memory

# Plot accuracies or any other metrics in a similar manner
# Make sure to use a different file name for each plot to avoid overwriting

# Assuming you also modify the test function to return accuracies over epochs, you can plot them similarly
train_dataset = MNISTSuperpixels(root='dataset/MNIST/', train=True)
test_dataset = MNISTSuperpixels(root='dataset/MNIST/', train=False)

class GCNModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, 64)
        # self.fc1 = torch.nn.Linear(num_features, 64)
        self.conv2 = GCNConv(64, 128)
        # self.fc2 = torch.nn.Linear(64, 128)
        self.conv3 = GCNConv(128, 128)
        # self.fc3 = torch.nn.Linear(128, 128)
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Applying the first convolution and skip connection
        # skip1 = self.fc1(x)  # Compute skip connection first
        x = self.conv1(x, edge_index) # Apply non-linearity after conv

        # x = x + skip1  # Add skip connection result

        x = F.relu(x)

        # Second convolution and skip connection
        # skip2 = self.fc2(x)  # Compute skip connection from output of previous layer
        x = self.conv2(x, edge_index)  # Apply non-linearity after conv

        # x = x + skip2  # Add skip connection result

        x = F.relu(x)

        # Third convolution and skip connection
        # skip3 = self.fc3(x)  # Compute skip connection from output of previous layer
        x = self.conv3(x, edge_index)  # Apply non-linearity after conv

        # x = x + skip3  # Add skip connection result

        x = F.relu(x)

        x = global_mean_pool(x, batch)  # Apply pooling to summarize the graph features
        x = self.fc(x)
        return x

class ChebNetModel(torch.nn.Module):
    def __init__(self, num_features, num_classes, K=2):
        super(ChebNetModel, self).__init__()
        self.K = 2
        self.conv1 = ChebConv(num_features, 64, self.K)
        self.conv2 = ChebConv(64, 128, self.K)
        self.conv3 = ChebConv(128, 128 , self.K)
        self.pool = global_mean_pool
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # No need to pass num_nodes to ChebConv layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.pool(x, batch)  # Pooling
        x = self.fc(x)  # Final fully connected layer
        return x

def train(model, train_loader, optimizer, criterion, device, epochs=50):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {epoch_loss}')
    return train_losses

def test(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data).max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print(f'Accuracy: {accuracy}')
    return accuracy


train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

num_features = train_dataset.num_features
num_classes = 10

gcn_model = GCNModel(num_features, num_classes).to('cuda')
chebnet_model = ChebNetModel(num_features, num_classes, K=2).to('cuda')

criterion = torch.nn.CrossEntropyLoss()
gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.001)
chebnet_optimizer = torch.optim.Adam(chebnet_model.parameters(), lr=0.001)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize and train GCN Model
gcn_losses = train(gcn_model, train_loader, gcn_optimizer, criterion, device, epochs=2000)
gcn_accuracy = test(gcn_model, test_loader, device)

# Initialize and train ChebNet Model
chebnet_losses = train(chebnet_model, train_loader, chebnet_optimizer, criterion, device, epochs=2000)
chebnet_accuracy = test(chebnet_model, test_loader, device)

import matplotlib.pyplot as plt

# Ensure the matplotlib backend is set up correctly in headless environments
plt.switch_backend('agg')

# Plot training losses and save the figure
plt.figure(figsize=(12, 6))
plt.plot(gcn_losses, label='GCN without skip connections Loss')
plt.plot(chebnet_losses, label='ChebNet Loss')
plt.title('Training Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss_comparison_without_skip.png')  # Saves the plot to a file
plt.close()  # Close the figure to free memory