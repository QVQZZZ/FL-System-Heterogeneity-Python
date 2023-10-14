import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, width_factor=1.0):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=int(6 * width_factor), kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=int(6 * width_factor), out_channels=int(16 * width_factor),
                               kernel_size=(5, 5))
        self.fc1 = nn.Linear(int(16 * width_factor * 4 * 4), int(120 * width_factor))
        self.fc2 = nn.Linear(int(120 * width_factor), int(84 * width_factor))
        self.fc3 = nn.Linear(int(84 * width_factor), 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features  # return (height * width * deepth)


def create_client_model(client_type, num_classes=10):
    if client_type == 'weak':
        return SimpleCNN(num_classes, width_factor=0.25)
    elif client_type == 'medium':
        return SimpleCNN(num_classes, width_factor=0.5)
    elif client_type == 'strong':
        return SimpleCNN(num_classes, width_factor=1.0)
    else:
        raise ValueError("Invalid client type")


num_clients = 5  # 客户端总数
clients_per_round = 2  # 每轮训练客户端比例
local_epochs = 5  # 本地迭代次数
total_epochs = 2  # 总迭代次数

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST(root="./data", train=True, download=False, transform=transform) # len == 60000
test_set = datasets.MNIST(root="./data", train=False, download=False, transform=transform) # len == 10000

# 划分客户端数据
client_data = torch.utils.data.random_split(train_set, [len(train_set) // num_clients] * num_clients)


def federated_learning(clients, total_epochs, local_epochs, clients_per_round):
    global_model = SimpleCNN()  # 创建全局模型

    for epoch in range(total_epochs):
        random.shuffle(clients)  # 随机选择一些客户端
        selected_clients = clients[:clients_per_round]

        for client in selected_clients:
            client_model = create_client_model(client['type'])  # 创建客户端特定宽度的模型
            client_data_loader = torch.utils.data.DataLoader(client['data'], batch_size=64, shuffle=True)
            optimizer = optim.SGD(client_model.parameters(), lr=0.01)

            for local_epoch in range(local_epochs):
                for batch_data, batch_labels in client_data_loader:
                    optimizer.zero_grad()
                    predictions = client_model(batch_data)
                    loss = nn.CrossEntropyLoss()(predictions, batch_labels)
                    loss.backward()
                    optimizer.step()

            # 更新全局模型
            for global_param, client_param in zip(global_model.parameters(), client_model.parameters()):
                global_param.data += client_param.data

        # 平均所有客户端的更新
        for global_param in global_model.parameters():
            global_param.data /= len(selected_clients)

        print(f"Epoch {epoch + 1}/{total_epochs} completed")

    return global_model


clients = [{'type': 'weak', 'data': client_data[i]} for i in range(num_clients)]
final_global_model = federated_learning(clients, total_epochs, local_epochs, clients_per_round)
