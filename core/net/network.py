import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, class_num):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 7, padding = (3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 7, padding = (3, 3))
        self.conv3 = nn.Conv2d(16, 32, 7, padding = (3, 3))
        self.conv4 = nn.Conv2d(32, 64, 7, padding=(3, 3))
        self.fc1 = nn.Linear(64 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 120)
        self.fc3 = nn.Linear(120, class_num)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

