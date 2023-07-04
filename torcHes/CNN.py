import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt 
import torch.nn.functional as F
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

num_epochs = 100
batch_size = 10
learning_rate = 0.045

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainDs = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testDs = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainDs, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testDs, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'dog', 'cat', 'deer', 'frog', 'horse', 'ship', 'truck')

#
class ConvNet(nn.Module) : 
    def __init__(self) :
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2 ,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x) :
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs) :
    for i ,(images, labels) in enumerate(train_loader) :
        images=images.to(device)
        labels=labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1)%200 == 0 :
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}, Loss : {loss.item()}]")

print("Finished Training")

with torch.no_grad() :
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader :
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size) : 
            label = labels[i]
            pred = predicted[i]
            if (label == pred) :
                n_class_correct[label] += 1
            n_class_samples[label] += 1


    for i in range(10) :
        acc = 100 * n_class_correct[i] / n_class_samples[i]
        print(f"Accuracy of {classes[i]} : {acc}%")
    acc = 100 * n_correct / n_samples
    print(f"OVRAccuracy : {acc}%")