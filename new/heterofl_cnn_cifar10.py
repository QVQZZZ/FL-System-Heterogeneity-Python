import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tool import cut_to_size, zeropad_to_size, dirichlet_split_noniid


class Factor:
    """
    Factor.width_factor返回列表形式的所有宽度的可能取值
    """
    width_factors = [0.25, 0.5, 1]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, width_factor=1.0):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=int(64 * width_factor), kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=int(64 * width_factor), out_channels=int(128 * width_factor), kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=int(128 * width_factor), out_channels=int(128 * width_factor), kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(int(128 * width_factor * 4 * 4), int(512 * width_factor))
        self.fc2 = nn.Linear(int(512 * width_factor), int(256 * width_factor))
        self.fc3 = nn.Linear(int(256 * width_factor), num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Client:
    def __init__(self, width_factor, data):
        self.width_factor = width_factor
        self.data = data

    def __str__(self):
        return f"Model(width_factor={self.width_factor}, data_len={len(self.data)})"


class Model:
    """
    Model类
    成员变量:
        width_factor:模型的尺寸
        parameters:真正的模型,是一个nn.Module实现类的实例
    """

    def __init__(self, width_factor, parameters):
        self.width_factor = width_factor
        self.parameters = parameters

    def __str__(self):
        return f"Model(width_factor={self.width_factor}, parameters={self.parameters})"


def init_global_model():
    """
    初始化width_factor=1的全局模型
    :return: 一个初始化后的width_factor=1的模型
    """
    parameters = SimpleCNN(num_classes=10, width_factor=1.0).cuda()
    width_factor = 1.0
    global_model = Model(width_factor=width_factor, parameters=parameters)
    return global_model


def random_choice(clients, clients_per_round):
    """
    从clients中采样clients_per_round个客户端,且客户端覆盖所有可能的width_factors,
    避免采样不到width_factor=1的客户端,导致模型聚合时出现大量的0
    :param clients: Client类组成的列表
    :param clients_per_round: 每轮通信选取的客户端数量
    :return: Client类组成的列表
    """
    while True:
        random.shuffle(clients)
        selected_clients = clients[:clients_per_round]
        clients_type = set([client.width_factor for client in selected_clients])
        if len(clients_type) == len(Factor.width_factors):
            return selected_clients


def create_client_model(width_factor):
    """
    HETEROFL: 根据客户端宽度创建一个客户端特定的模型
    :param width_factor: 客户端宽度
    :return: 一个客户端特定的模型
    """
    parameters = SimpleCNN(num_classes=10, width_factor=width_factor).cuda()
    client_model = Model(width_factor=width_factor, parameters=parameters)
    return client_model


def get_parameters_from_server(client_model, global_model):
    """
    从服务器获取参数
    :param client_model: 客户端网络
    :param global_model: 服务器网络
    :return: 从服务器网络获取参数后的客户端网络
    """
    global_model_dict = global_model.parameters.state_dict()
    client_model_dict = client_model.parameters.state_dict()
    for name, param in client_model_dict.items():
        if name in global_model_dict.keys():
            client_model_dict[name] = cut_to_size(global_model_dict[name], client_model_dict[name].size())
    client_model.parameters.load_state_dict(client_model_dict)
    return client_model


def aggregate(received_models):
    """
    聚合模型得到一个全局模型
    :param received_models: 多个客户端模型组成的列表
    :return: 聚合后的全局模型
    """
    # zero init global_model(aggregated_model)
    global_model = SimpleCNN(10, 1.0).cuda()
    global_model.load_state_dict(
        {name: nn.init.zeros_(param) for name, param in global_model.named_parameters()}
    )
    global_dict = {name: param for name, param in global_model.named_parameters()}

    # sum up
    for model_object in received_models:
        model = model_object.parameters
        named_parameters_generator = model.named_parameters()
        for name, val in named_parameters_generator:
            val = zeropad_to_size(val, global_dict[name].size())
            with torch.no_grad():
                global_dict[name].add_(val)

    # weight average
    from collections import Counter
    width_factors = [model.width_factor for model in received_models]
    counter = Counter(width_factors)
    sorted_count = sorted(counter.items())
    distribution = [item[1] for item in sorted_count]
    weight_to_divide = [sum(distribution[i:]) for i in range(len(distribution))]
    weight_to_multiply = [1 / x for x in weight_to_divide]
    # print('width_factors', width_factors)
    # print('distribution', distribution)
    # print('weight_to_multiply', weight_to_multiply)

    temp = {}
    for name, val in global_dict.items():
        coefficients = torch.ones(val.size()).to('cuda')
        if 'conv' in name:
            conv_number = val.size()[0]
            coefficients[:int(0.25 * conv_number)] = weight_to_multiply[0]
            coefficients[int(0.25 * conv_number):int(0.5 * conv_number)] = weight_to_multiply[1]
            coefficients[int(0.5 * conv_number):] = weight_to_multiply[2]
        if 'fc' in name and 'weight' in name and 'last' not in name:
            dim1, dim2 = val.size()[0], val.size()[1]
            coefficients[:int(0.25 * dim1), :int(0.25 * dim2)] = weight_to_multiply[0]
            coefficients[int(0.25 * dim1):int(0.5 * dim1), :int(0.5 * dim2)] = weight_to_multiply[1]
            coefficients[:int(0.25 * dim1), int(0.25 * dim2):int(0.5 * dim2)] = weight_to_multiply[1]
            coefficients[int(0.5 * dim1):, :] = weight_to_multiply[2]
            coefficients[:int(0.5 * dim1), int(0.5 * dim2):] = weight_to_multiply[2]
        if 'fc' in name and 'bias' in name and 'last' not in name:
            dim = val.size()[0]
            coefficients[:int(0.25 * dim)] = weight_to_multiply[0]
            coefficients[int(0.25 * dim):int(0.5 * dim)] = weight_to_multiply[1]
            coefficients[int(0.5 * dim):] = weight_to_multiply[2]
        if 'fc' in name and 'weight' in name and 'last' in name:
            dim = val.size()[1]
            coefficients[:, :int(0.25 * dim)] = weight_to_multiply[0]
            coefficients[:, int(0.25 * dim):int(0.5 * dim)] = weight_to_multiply[1]
            coefficients[:, int(0.5 * dim):] = weight_to_multiply[2]
        if 'fc' in name and 'bias' in name and 'last' in name:
            coefficients[:] = 1 / 3
        val = val * coefficients
        temp[name] = val
    global_dict = temp

    global_model.load_state_dict(global_dict)
    return Model(width_factor=1.0, parameters=global_model)


def heterofl(clients, clients_per_round, total_epochs, local_epochs):
    global_model = init_global_model()
    for total_epoch in range(total_epochs):
        clients_selected = random_choice(clients, clients_per_round)
        received_models = []
        for client in clients_selected:
            client_model = create_client_model(client.width_factor)
            client_model = get_parameters_from_server(client_model, global_model)
            # backward start
            client_data_loader = torch.utils.data.DataLoader(client.data, batch_size=64, shuffle=True)
            optimizer = torch.optim.Adam(client_model.parameters.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            for local_epoch in range(local_epochs):
                for batch_data, batch_labels in client_data_loader:
                    client_model.parameters.train()  # train mode
                    batch_data, batch_labels = batch_data.to('cuda'), batch_labels.to('cuda')
                    optimizer.zero_grad()
                    predictions = client_model.parameters(batch_data)
                    loss = criterion(predictions, batch_labels)
                    loss.backward()
                    optimizer.step()
            # backward end
            received_models.append(client_model)
        global_model = aggregate(received_models)
        print(f"Epoch {total_epoch + 1}/{total_epochs} completed")

        test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
        accuracy = test_model(global_model.parameters.to('cuda'), test_data_loader)
        print(f'Accuracy on the test set: {accuracy:.2f}%')
    return global_model


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to('cuda'), labels.to('cuda')  # 将测试数据移到GPU
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


if __name__ == '__main__':
    # prepare dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)  # len == 60000
    test_set = datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)  # len == 10000

    # split data into different clients (iid)
    # num_clients = 10
    # client_data = random_split(train_set, [len(train_set) // num_clients] * num_clients)
    # client_width_factors = np.random.choice(Factor.width_factors, num_clients) # Factor.width_factors = [0.25,0.5,1]
    # clients = [Client(width_factor=width_factor, data=data) for width_factor, data in zip(client_width_factors, client_data)]

    # split data into different clients (non-iid)
    num_clients = 10
    client_width_factors = np.random.choice(Factor.width_factors, num_clients)
    client_idcs = dirichlet_split_noniid(np.array(train_set.targets), alpha=1, n_clients=num_clients)
    clients = []
    for client_idx in range(num_clients):
        client_indices = client_idcs[client_idx]
        client_data = [train_set[i] for i in client_indices]
        width_factor = client_width_factors[client_idx]
        clients.append(Client(width_factor=width_factor, data=client_data))

    # starting federated learning
    selected_rate = 0.5
    final_global_model = heterofl(clients=clients, clients_per_round=int(selected_rate * num_clients), total_epochs=5,
                                  local_epochs=3).parameters

    # 创建测试数据加载器
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    # 测试 final_global_model
    accuracy = test_model(final_global_model.to('cuda'), test_data_loader)
    print('Training completed')
    print(f'Accuracy on the test set: {accuracy:.2f}%')
