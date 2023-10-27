import torch.nn.functional as F
import torch.nn as nn


class CifarCNN(nn.Module):
    def __init__(self, num_classes=10, width_factor=1.0):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=int(64 * width_factor), kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=int(64 * width_factor), out_channels=int(128 * width_factor), kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=int(128 * width_factor), out_channels=int(128 * width_factor), kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(int(128 * width_factor * 4 * 4), int(512 * width_factor))
        self.fc2 = nn.Linear(int(512 * width_factor), int(256 * width_factor))
        self.fc3_last = nn.Linear(int(256 * width_factor), num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3_last(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class MnistCNN(nn.Module):
    def __init__(self, num_classes=10, width_factor=1.0):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=int(6 * width_factor), kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=int(6 * width_factor), out_channels=int(16 * width_factor), kernel_size=(5, 5))
        self.fc1 = nn.Linear(int(16 * width_factor * 4 * 4), int(120 * width_factor))
        self.fc2 = nn.Linear(int(120 * width_factor), int(84 * width_factor))
        self.fc3_last = nn.Linear(int(84 * width_factor), num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3_last(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
