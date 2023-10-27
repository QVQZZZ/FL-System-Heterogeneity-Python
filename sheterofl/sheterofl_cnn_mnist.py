# In[import and tool]
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def zeropad_to_size(small_tensor, big_tensor_size):
    diff = np.array(big_tensor_size) - np.array(small_tensor.size())
    #  if 'diff' is: np.array([20,10,0,0])
    #  then pad is: (0,0,0,0,0,10,0,20)
    pad = []
    for x in diff[::-1]:
        if x == 0:
            pad.append(0)
            pad.append(0)
        else:
            pad.append(0)
            pad.append(x)
    pad = tuple(pad)
    return F.pad(small_tensor, pad, 'constant', 0)


def cut_to_size(big_tensor, small_tensor_size):
    diff = np.array(small_tensor_size) - np.array(big_tensor.size())
    #  if 'diff' is: np.array([-10,-5,0,0])
    #  then pad is: (0,0,0,0,0,-5,0,-10)
    pad = []
    for x in diff[::-1]:
        if x == 0:
            pad.append(0)
            pad.append(0)
        else:
            pad.append(0)
            pad.append(x)
    pad = tuple(pad)
    return F.pad(big_tensor, pad, 'constant', 0)

# In[net]
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, width_factor=1.0):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=int(6 * width_factor), kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=int(6 * width_factor), out_channels=int(16 * width_factor), kernel_size=(5, 5))
        self.fc1 = nn.Linear(int(16 * width_factor * 4 * 4), int(120 * width_factor))
        self.fc2 = nn.Linear(int(120 * width_factor), int(84 * width_factor))
        self.fc3 = nn.Linear(int(84 * width_factor), num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# In[federated learning: heteroFL]
def create_client_model(client_type, num_classes=10):
    if client_type == 'weak':
        return SimpleCNN(num_classes, width_factor=0.25)
    elif client_type == 'medium':
        return SimpleCNN(num_classes, width_factor=0.5)
    elif client_type == 'strong':
        return SimpleCNN(num_classes, width_factor=1.0)
    else:
        raise ValueError("Invalid client type")




def federated_learning(clients, clients_per_round, total_epochs, local_epochs):
    global_model = SimpleCNN().to('cuda')

    for epoch in range(total_epochs):
        random.shuffle(clients)
        selected_clients = clients[:clients_per_round]
        clients_model_parameters_collections = {'weak': [], 'medium': [], 'strong': []}
        clients_type_counts_collections = {'weak': 0, 'medium': 0, 'strong': 0}
        aggregated_params = {param_name: torch.zeros_like(param) for param_name, param in global_model.named_parameters()}


        for client in selected_clients:
            # TODO: SHETEROFL
            # Create client-specific width model
            client_models = []  # [('strong',NN), ('medium',NN), ...]
            if client['type'] == 'strong':
                client_models.append(('strong', create_client_model('strong').to('cuda')))
                client_models.append(('medium', create_client_model('medium').to('cuda')))
                client_models.append(('weak', create_client_model('weak').to('cuda')))
            elif client['type'] == 'medium':
                client_models.append(('medium', create_client_model('medium').to('cuda')))
                client_models.append(('weak', create_client_model('weak').to('cuda')))
            else:  # weak
                client_models.append(('weak', create_client_model('weak').to('cuda')))

            for type, client_model in client_models:
                #  Client get model parameters from server
                global_model_dict = global_model.state_dict()
                client_model_dict = client_model.state_dict()
                for name, param in client_model_dict.items():
                    if name in global_model_dict.keys():
                        client_model_dict[name] = cut_to_size(global_model_dict[name], client_model_dict[name].size())
                client_model.load_state_dict(client_model_dict)

                client_data_loader = torch.utils.data.DataLoader(client['data'], batch_size=64, shuffle=True)
                optimizer = torch.optim.Adam(client_model.parameters())
                criterion = torch.nn.CrossEntropyLoss()

                for local_epoch in range(local_epochs):
                    for batch_data, batch_labels in client_data_loader:
                        batch_data, batch_labels = batch_data.to('cuda'), batch_labels.to('cuda')
                        optimizer.zero_grad()
                        predictions = client_model(batch_data)
                        loss = criterion(predictions, batch_labels)
                        loss.backward()
                        optimizer.step()

                #  在SHETEROFL下，客户端可以有多个model，因此以下的变量可以理解为model的type
                #  例如，原本是强客户端、弱客户端的统计，现在可以理解为强模型、弱模型的统计，也因此上面client_models会附带字符串作为元组
                #  原本需要将变量改名，但是太麻烦了
                #  确实可以写的很优雅，就是让model和client['type']一样加上一个type属性，但是我懒，就这样用if语句得了
                clients_model_parameters_collections[type].append(client_model.named_parameters()) #  names_parameters() returns (name, val)
                clients_type_counts_collections[type] += 1

                # Server aggregates and average
                clients_parameters_weight_collections = {key: value / clients_per_round for key, value in clients_type_counts_collections.items()}
                for client_type, named_parameters_list in clients_model_parameters_collections.items():
                    for named_parameters in named_parameters_list:
                        for name, val in named_parameters:
                            val = zeropad_to_size(val, aggregated_params[name].size())
                            aggregated_params[name].add_(val * clients_parameters_weight_collections[client_type])

        # Load the aggregated parameters into the global model
        global_model.load_state_dict(aggregated_params)

        print(f"Epoch {epoch + 1}/{total_epochs} completed")

    return global_model


# In[train and test: heteroFL]
num_clients = 20  # 客户端总数
clients_per_round = 5  # 每轮训练客户端比例
local_epochs = 2  # 本地迭代次数
total_epochs = 10  # 总迭代次数

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST(root="../data", train=True, download=True, transform=transform)  # len == 60000
test_set = datasets.MNIST(root="../data", train=False, download=True, transform=transform)  # len == 10000

# 划分客户端数据
client_data = random_split(train_set, [len(train_set) // num_clients] * num_clients)
client_types = ['weak', 'medium', 'strong']
client_type_distribution = np.random.choice(client_types, num_clients)
clients = [{'type': client_type, 'data': client_data[i]} for i, client_type in enumerate(client_type_distribution)]
final_global_model = federated_learning(clients, clients_per_round, total_epochs, local_epochs)


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

# 创建测试数据加载器
test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# 测试 final_global_model
accuracy = test_model(final_global_model.to('cuda'), test_data_loader)
print(f'Accuracy on the test set: {accuracy:.2f}%')