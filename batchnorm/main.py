import torch
import torchvision
import torch.optim as optim
import torch.nn as nn 
import torchvision.transforms as transforms
import torch.nn.functional as F 

cls_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(
                                        root='data', 
                                        train=True, 
                                        download=True, 
                                        transform=cls_transform
                                        )

testset = torchvision.datasets.CIFAR10(
                                        root='data',
                                        train=False,
                                        download=True,
                                        transform=cls_transform
                                        )

batch_size = 4

train_loader = torch.utils.data.DataLoader(
                                        dataset=trainset,
                                        batch_size=batch_size,
                                        shuffle=True
                                        )

test_loader = torch.utils.data.DataLoader(
                                        dataset=testset,
                                        batch_size=batch_size,
                                        shuffle=False
                                        )

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
    
net = Net()
