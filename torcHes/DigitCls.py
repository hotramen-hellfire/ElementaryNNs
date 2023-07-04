import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib as plt

device = torch.device('cpu')

input_size = 784
hidden_size = 128
num_classes = 10
num_epochs = 3
batch_size = 100
learning_rate = 0.01

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

model = NeuralNet(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs) : 
    for i, (images, labels) in enumerate(train_loader) : 
        
        images = images.reshape(-1, 784).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0 :
            print(f"epoch {epoch+1} / {num_epochs}, step{i+1}/{n_total_steps}, loss={loss.item()}")

with torch.no_grad() :
    n_correct=0
    n_samples=0
    for images, labels in test_loader : 
        images = images.reshape(-1, 784).to(device)
        labels=labels = labels.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs, 1)
        n_samples+=labels.shape[0]
        n_correct+=(predictions == labels).sum().item()

    acc = 100 * n_correct / n_samples
    print(f"accuracy = {acc}")