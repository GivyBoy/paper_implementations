import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes=10, use_bn=True):
        super(CNN, self).__init__()
        self.use_bn = use_bn

        # First conv block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(32) if use_bn else nn.Identity()

        # Second conv block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(64) if use_bn else nn.Identity()

        # Third conv block
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=not use_bn)
        self.bn3 = nn.BatchNorm2d(64) if use_bn else nn.Identity()

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)

        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
