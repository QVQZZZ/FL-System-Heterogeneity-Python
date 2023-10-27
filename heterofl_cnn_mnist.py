# In[0]
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import random_split

from tool import cut_to_size, zeropad_to_size, dirichlet_split_noniid
from datasets import load_dataset
from metrics import test_model
from fl import Factor, Client, Model
from models import MnistCNN as SimpleCNN


def init_global_model():
    """
    初始化width_factor=1的全局模型
    :return: 一个初始化后的width_factor=1的模型
    """
    parameters = SimpleCNN(num_classes=10, width_factor=1.0).cuda()
    global_model = Model(width_factor=1.0, parameters=parameters)
    return global_model


def create_client_model(width_factor):
    """
    HETEROFL: 根据客户端宽度创建一个客户端特定的模型
    :param width_factor: 客户端宽度
    :return: 一个客户端特定的模型
    """
    parameters = SimpleCNN(num_classes=10, width_factor=width_factor).cuda()
    client_model = Model(width_factor=width_factor, parameters=parameters)
    return client_model


def random_choice(clients, clients_per_round, difference=True):
    """
    从clients中采样clients_per_round个客户端,且客户端覆盖所有可能的width_factors,
    避免采样不到width_factor=1的客户端,导致模型聚合时出现大量的0
    :param clients: Client类组成的列表
    :param clients_per_round: 每轮通信选取的客户端数量
    :param difference: 是否保证选出的客户端包含所有width_factors
    :return: Client类组成的列表
    """
    if difference:
        while True:
            random.shuffle(clients)
            selected_clients = clients[:clients_per_round]
            clients_type = set([client.width_factor for client in selected_clients])
            if len(clients_type) == len(Factor.width_factors):
                return selected_clients
    else:
        random.shuffle(clients)
        selected_clients = clients[:clients_per_round]
        return selected_clients


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
    model_class = type(received_models[0].parameters)
    global_model = model_class(num_classes=10, width_factor=1.0).cuda()
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
    # 以下用来处理缺少width_factors的情况,例如只有0.5和1.0而没有0.25,可能还有BUG
    diff_width_factor = np.array(Factor.width_factors)[~np.isin(np.array(Factor.width_factors), np.unique(np.array(width_factors)))]  # 缺少的width_factor
    for x in diff_width_factor:
        sorted_count.append((x, 0))
    sorted_count = sorted(sorted_count)
    # 处理缺少width_factors结束
    distribution = [item[1] for item in sorted_count]
    assert distribution[-1] != 0  # 确保width_factor=1的客户端的存在,否则聚合可能导致大面积的0
    weight_to_divide = [sum(distribution[i:]) for i in range(len(distribution))]
    weight_to_multiply = [1 / x for x in weight_to_divide]

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
            coefficients[:] = 1 / len(received_models)
        val = val * coefficients
        temp[name] = val
    global_dict = temp

    global_model.load_state_dict(global_dict)
    return Model(width_factor=1.0, parameters=global_model)


def heterofl(clients, clients_per_round, total_epochs, local_epochs, difference=True):
    global_model = init_global_model()
    for total_epoch in range(total_epochs):
        clients_selected = random_choice(clients, clients_per_round, difference=difference)
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


# In[1]
if __name__ == '__main__':
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    cfg = {"num_clients": 100,
           "selected_rate": 0.1,
           "total_epoch": 30,
           "local_epoch": 3,
           # 是否保证每轮联邦选出的客户端包含所有width_factors,一般选择True,如果你在测试Exclusive,test_small_control等
           # 可能造成缺少某种客户端的情况,那么需要将difference改为False
           "difference": False,
           # 可选为iid, dirichlet, test_small_exp, test_small_control
           "split_method": "test_small_control",
           }

    # prepare dataset
    train_set, test_set = load_dataset(path="./data", name="mnist")

    num_clients = cfg["num_clients"]
    client_width_factors = np.random.choice(Factor.width_factors, num_clients)
    client_width_factors = np.sort(client_width_factors)  # 便于split_method为test_small时划分数据集

    if cfg["split_method"] == "iid":
        # split data into different clients (iid)
        client_data = random_split(train_set, [len(train_set) // num_clients] * num_clients)
        clients = [Client(width_factor=width_factor, data=data) for width_factor, data in zip(client_width_factors, client_data)]
    elif cfg["split_method"] == "dirichlet":
        # split data into different clients (dirichlet noniid)
        client_index = dirichlet_split_noniid(np.array(train_set.targets), alpha=1, n_clients=num_clients)
        client_data = [Subset(train_set, indices) for indices in client_index]
        clients = [Client(width_factor=width_factor, data=data) for width_factor, data in zip(client_width_factors, client_data)]
    elif cfg["split_method"] == "test_small_exp":
        # split data into different clients (test small model)
        # 将0,1,2的样本dirichlet划分到小客户端上, 将剩余的样本dirichlet划分到其他客户端上
        small_client = np.where(client_width_factors == min(client_width_factors))[0]
        other_client = np.where(client_width_factors != min(client_width_factors))[0]
        dataset_part1 = Subset(train_set, [i for i in range(len(train_set)) if train_set.targets[i] in [0, 1, 2]])
        dataset_part2 = Subset(train_set, [i for i in range(len(train_set)) if train_set.targets[i] not in [0, 1, 2]])
        dataset_part1_targets = [train_set[i][1] for i in dataset_part1.indices]  # Subset对象没有targets属性, 需要手动创建
        dataset_part2_targets = [train_set[i][1] for i in dataset_part2.indices]  # Subset对象没有targets属性, 需要手动创建
        small_client_index = dirichlet_split_noniid(np.array(dataset_part1_targets), alpha=1, n_clients=len(small_client), subset_indices=np.array(dataset_part1.indices))
        small_client_data = [Subset(train_set, indices) for indices in small_client_index]
        other_client_index = dirichlet_split_noniid(np.array(dataset_part2_targets), alpha=1, n_clients=len(other_client), subset_indices=np.array(dataset_part2.indices))
        other_client_data = [Subset(train_set, indices) for indices in other_client_index]
        client_data = small_client_data + other_client_data
        clients = [Client(width_factor=width_factor, data=data) for width_factor, data in zip(client_width_factors, client_data)]
    elif cfg["split_method"] == "test_small_control":
        # split data into different clients (test small model)
        # 将0,1,2的样本dirichlet划分到小客户端上, 将剩余的样本dirichlet划分到其他客户端上
        # 随后去掉small客户端
        small_client = np.where(client_width_factors == min(client_width_factors))[0]
        other_client = np.where(client_width_factors != min(client_width_factors))[0]
        dataset_part1 = Subset(train_set, [i for i in range(len(train_set)) if train_set.targets[i] in [0, 1, 2]])
        dataset_part2 = Subset(train_set, [i for i in range(len(train_set)) if train_set.targets[i] not in [0, 1, 2]])
        dataset_part1_targets = [train_set[i][1] for i in dataset_part1.indices]  # Subset对象没有targets属性, 需要手动创建
        dataset_part2_targets = [train_set[i][1] for i in dataset_part2.indices]  # Subset对象没有targets属性, 需要手动创建
        small_client_index = dirichlet_split_noniid(np.array(dataset_part1_targets), alpha=1, n_clients=len(small_client), subset_indices=np.array(dataset_part1.indices))
        small_client_data = [Subset(train_set, indices) for indices in small_client_index]
        other_client_index = dirichlet_split_noniid(np.array(dataset_part2_targets), alpha=1, n_clients=len(other_client), subset_indices=np.array(dataset_part2.indices))
        other_client_data = [Subset(train_set, indices) for indices in other_client_index]
        client_data = small_client_data + other_client_data
        clients = [Client(width_factor=width_factor, data=data) for width_factor, data in zip(client_width_factors, client_data)]
        clients = [_ for _ in clients if _.width_factor != min(Factor.width_factors)]
    else:
        raise ValueError("Invalid split method type")

    # starting federated learning
    selected_rate, total_epoch, local_epoch, difference = cfg["selected_rate"], cfg["total_epoch"], cfg["local_epoch"], cfg["difference"]
    final_global_model = heterofl(clients=clients, clients_per_round=int(selected_rate*num_clients),
                                  total_epochs=total_epoch, local_epochs=local_epoch, difference=difference).parameters

    # test final_global_model
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    accuracy = test_model(final_global_model.to('cuda'), test_data_loader)
    print('Training completed')
    print(f'Accuracy on the test set: {accuracy:.2f}%')
