import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import time

# Function to load and preprocess the MNIST dataset.
def load_mnist_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

# Function to create the neural network model.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Function to run the training loop.
def train_network(net, optimizer, train_loader, num_epochs, device):
    for epoch in tqdm(range(num_epochs)):
        net.train()
        total_correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data.to(device)
            output = net(data)
            output.to('cpu')
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

            total_correct += (output.argmax(1) == target).type(torch.float).sum().item()
            total += data.size(0)
        
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        print(f"Train accuracy: {total_correct / total}")

# Function to evaluate the network on the test dataset.
def evaluate_network(net, test_loader, device):
    net.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data.to(device)
            output = net(data)
            output.to('cpu')
            total_correct += (output.argmax(1) == target).type(torch.float).sum().item()
            total += data.size(0)

    return total_correct / total

# Main execution
def main():
    train_loader, test_loader = load_mnist_data()
    net = NeuralNetwork()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    num_epochs = 10
    device = 'cpu'
    net.to(device)

    start_time = time.time()
    train_network(net, optimizer, train_loader, num_epochs, device)
    elapsed = time.time() - start_time
    print(f"Training time: {elapsed}")
    
    test_accuracy = evaluate_network(net, test_loader, device)
    print(f"Test accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
