# In[0]
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
import copy
from typing import List

from tool import *
from datasets import *
from metrics import *
from fl import *


def init_global_model():
    """
    初始化width_factor=1的全局模型
    :return: 一个初始化后的width_factor=1的模型
    """
    parameters = Net(num_classes=10, width_factor=1.0).cuda()
    global_model = Model(width_factor=1.0, parameters=parameters)
    return global_model


def create_client_model(width_factor, multiple=False):
    """
    根据客户端宽度创建一个客户端特定的模型
    :param width_factor: 客户端宽度
    :param multiple: 若为True则会返回多个宽度小于客户端宽度的模型组成的列表
    :return: 一个客户端特定的模型,或一个模型的列表
    """
    if not multiple:
        parameters = Net(num_classes=10, width_factor=width_factor).cuda()
        client_model = Model(width_factor=width_factor, parameters=parameters)
        return client_model
    else:
        width_factors = np.array(Factor.width_factors)
        width_factors = width_factors[width_factors <= width_factor]
        client_models = []
        for width_factor in width_factors:
            parameters = Net(num_classes=10, width_factor=width_factor).cuda()
            client_model = Model(width_factor=width_factor, parameters=parameters)
            client_models.append(client_model)
        return client_models


def random_choice(clients, clients_per_round, difference=True):
    """
    从clients中采样clients_per_round个客户端,且客户端覆盖所有可能的width_factors,
    避免采样不到width_factor=1的客户端,导致模型聚合时出现大量的0
    :param clients: Client类组成的列表
    :param clients_per_round: 每轮通信选取的客户端数量
    :param difference: 是否保证选出的客户端包含所有width_factors
    :return: (Client类组成的列表, 选中客户端在原始列表中的索引)
    """
    if difference:
        # 若为True,则保证选出的客户端包含所有宽度的客户端
        while True:
            shuffled_clients = random.sample(clients, len(clients))
            selected_clients = shuffled_clients[:clients_per_round]
            clients_type = set([client.width_factor for client in selected_clients])
            if len(clients_type) == len(Factor.width_factors):
                idx = [clients.index(selected_client) for selected_client in selected_clients]
                return selected_clients, idx
    else:
        # 若为False,也至少要保证选出的客户端中包含一个宽度为1的客户端,避免聚合时出现大量的0
        while True:
            shuffled_clients = random.sample(clients, len(clients))
            selected_clients = shuffled_clients[:clients_per_round]
            for selected_client in selected_clients:
                if selected_client.width_factor == 1:
                    idx = [clients.index(selected_client) for selected_client in selected_clients]
                    return selected_clients, idx


class KnowledgeDistillationLossWithRegularization(nn.Module):
    def __init__(self, temperature=1, lambda_reg=0.01):
        super(KnowledgeDistillationLossWithRegularization, self).__init__()
        self.temperature = temperature
        self.lambda_reg = lambda_reg
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs, teacher_outputs, model, old_model):
        outputs = self.log_softmax(outputs / self.temperature)
        teacher_outputs = self.softmax(teacher_outputs / self.temperature)
        loss_kd = self.kldiv_loss(outputs, teacher_outputs)

        l2_regularization = 0.0
        for param, old_param in zip(model.parameters(), old_model.parameters()):
            l2_regularization += torch.norm(param - old_param, p=2)
        # 总损失 = 知识蒸馏损失 + L2正则化项
        loss_total = loss_kd + self.lambda_reg * l2_regularization
        # print(f'loss_kd/loss_l2={loss_kd / (self.lambda_reg * l2_regularization)}')
        return loss_total


def distribute(global_model, client_models, server_data, kd_epochs):
    """
    根据服务器模型，蒸馏客户端模型，相当于客户端从服务器处获取参数
    :param global_model: 最新的服务器模型 (one teacher)
    :param client_models: 客户端模型组成的列表 (many student)
    :param server_data: 服务器上用于蒸馏的数据
    :param kd_epochs: 服务器模型对每个客户端模型蒸馏的 epoch 次数
    :return: 蒸馏出的客户端模型的列表
    """
    distribute_models = []
    for client_model in client_models:
        if client_model.width_factor < 1:
            distilled_model = kd2p(global_model, client_model.width_factor, client_model, torch.utils.data.ConcatDataset(server_data), kd_epochs, temperature=1, lambda_reg=0)
            distribute_models.append(distilled_model)
        else:
            distribute_models.append(copy.deepcopy(global_model))
    return distribute_models

# def distribute(new_server_model, client_models, server_data, kd_epochs):
#     """
#     根据最新蒸馏出来的服务器模型，对上一轮收到的客户端模型进行更新，返回给各个客户端
#     :param new_server_model: 最新的服务器模型 (one teacher)
#     :param client_models: 客户端模型组成的列表 (many student)
#     :param server_data: 服务器上用于蒸馏的数据
#     :param kd_epochs: 服务器模型对每个客户端模型蒸馏的 epoch 次数
#     :return: 蒸馏出的客户端模型的列表
#     """
#     distribute_models = []
#     for received_model, data in zip(client_models, server_data):
#         distillation_model = copy.deepcopy(received_model.parameters)  # 克隆接收到的模型用于蒸馏优化
#         distillation_model.train()
#         criterion = KnowledgeDistillationLossWithRegularization(temperature=3, lambda_reg=0.001)
#         optimizer = torch.optim.Adam(distillation_model.parameters(), lr=0.001)
#         dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
#
#         # 对该接收到的模型进行蒸馏优化
#         for epoch in range(kd_epochs):
#             for inputs, labels in dataloader:
#                 inputs, labels = inputs.to('cuda'), labels.to('cuda')
#                 optimizer.zero_grad()
#                 with torch.no_grad():
#                     outputs = new_server_model.parameters(inputs)
#                 received_outputs = distillation_model(inputs)
#                 loss = criterion(received_outputs, outputs, distillation_model, received_model.parameters)
#                 loss.backward()
#                 optimizer.step()
#
#         # 将蒸馏后的模型添加到列表中
#         distribute_models.append(Model(width_factor=received_model.width_factor, parameters=distillation_model))
#
#     return distribute_models  # 返回蒸馏后的模型列表


def aggregate(received_models: List[Model], global_model: Model, server_data: List[Subset], kd_epochs) -> Model:
    """
    根据所有接收到的模型，统一蒸馏为p=1的模型，然后做fedavg
    :param received_models: 接收到的所有模型组成的列表 (many teachers)
    :param global_model: 上一轮的服务器模型 (one student)
    :param server_data: 服务器上用于蒸馏的数据
    :param kd_epochs: 每个客户端对服务器模型蒸馏的 epoch 次数
    :return: 蒸馏出的服务器模型
    """
    # 创建p=1模型的平均作为正则化目标模型
    temp_state_dict = global_model.parameters.state_dict()
    pmax_models = [_ for _ in received_models if _.width_factor == max(Factor.width_factors)]
    for key in temp_state_dict:
        temp_state_dict[key] = torch.stack([pmax_models[i].parameters.state_dict()[key] for i in range(len(pmax_models))], dim=0).mean(0)
    temp_model = init_global_model()
    temp_model.parameters.load_state_dict(temp_state_dict)
    # 创建完成，若需要以该模型为目标，则kd2p的第三个参数从global_model改为temp_model


    models = []
    for received_model in received_models:
        if received_model.width_factor < 1:
            distilled_model = kd2p(received_model, 1.0, temp_model, torch.utils.data.ConcatDataset(server_data), kd_epochs, temperature=1, lambda_reg=0.1)
            models.append(distilled_model)
        else:
            models.append(received_model)

    # test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    # sub_accuracies = [test_model(model.parameters.to('cuda'), test_data_loader) for model in models]
    # print('after_kd')
    # print(sub_accuracies)

    with torch.no_grad():
        global_state_dict = global_model.parameters.state_dict()
        for key in global_state_dict:
            global_state_dict[key] = torch.stack([models[i].parameters.state_dict()[key] for i in range(len(models))], dim=0).mean(0)
        global_model.parameters.load_state_dict(global_state_dict)

    return global_model


def kd2p(model, dest_width_factor, last_model, server_data, kd_epochs, temperature=1, lambda_reg=0.1):
    """
    采用知识蒸馏将模型变成指定的width_factor大小
    :param model: 需要蒸馏的模型 (教师模型)
    :param dest_width_factor: 蒸馏后的模型的大小
    :param last_model: 蒸馏回去的时候需要保证与last_model接近，避免模型太过发散
    :param server_data: 服务器上用于蒸馏的数据
    :param kd_epochs: 蒸馏的epoch次数
    :param temperature: 蒸馏温度
    :param lambda_reg: 正则化强度
    :return: 蒸馏后的模型
    """
    # print(f'kd2p被调用,源{model.width_factor},目的{dest_width_factor}')
    dest_model = create_client_model(width_factor=dest_width_factor).parameters
    dest_model.train()
    criterion = KnowledgeDistillationLossWithRegularization(temperature=temperature, lambda_reg=lambda_reg)
    optimizer = torch.optim.Adam(dest_model.parameters())
    dataloader = torch.utils.data.DataLoader(server_data, batch_size=64, shuffle=True)
    for epoch in range(kd_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = dest_model(inputs)
            with torch.no_grad():
                teacher_outputs = model.parameters(inputs)
            loss = criterion(outputs, teacher_outputs, dest_model, last_model.parameters)
            loss.backward()
            optimizer.step()
    return Model(width_factor=dest_width_factor, parameters=dest_model)



def nicekd(clients, clients_per_round, total_epochs, local_epochs, difference=True):
    global_model = init_global_model()
    client_models = [create_client_model(client.width_factor) for client in clients]

    # create server data
    server_data = []
    for subset in [client.data for client in clients]:
        subset_length = len(subset)
        one_tenth = subset_length // 10
        new_indices = subset.indices[:one_tenth]
        new_subset = Subset(subset.dataset, new_indices)
        server_data.append(new_subset)
    # end create server data

    # 小小训练以下 global_model，避免冷启动问题
    client_data_loader = torch.utils.data.DataLoader(clients[0].data, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(global_model.parameters.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    for local_epoch in range(1):
        for batch_data, batch_labels in client_data_loader:
            global_model.parameters.train()  # train mode
            batch_data, batch_labels = batch_data.to('cuda'), batch_labels.to('cuda')
            optimizer.zero_grad()
            predictions = global_model.parameters(batch_data)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
    print('here')
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    print(test_model(global_model.parameters.to('cuda'), test_data_loader))


    for total_epoch in range(total_epochs):
        clients_selected, idx = random_choice(clients, clients_per_round, difference=difference)
        models_selected = [client_models[i] for i in idx]

        # print('before distribute')
        # test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
        # sub_accuracies = [test_model(received_model.parameters.to('cuda'), test_data_loader) for received_model in models_selected]
        # print(sub_accuracies)

        models_selected = distribute(global_model, models_selected, server_data, 5)

        # print('after distribute')
        # test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
        # sub_accuracies = [test_model(received_model.parameters.to('cuda'), test_data_loader) for received_model in models_selected]
        # print(sub_accuracies)

        for m, i in zip(models_selected, idx):
            client_models[i] = m

        received_models = []
        for client, client_model in zip(clients_selected, models_selected):
            # backward start
            client_data_loader = torch.utils.data.DataLoader(client.data, batch_size=64, shuffle=True)
            optimizer = torch.optim.Adam(client_model.parameters.parameters())
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
        global_model = aggregate(received_models, global_model, server_data, 5)

        print(f"Epoch {total_epoch + 1}/{total_epochs} completed")
        test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
        # sub_accuracies = [test_model(received_model.parameters.to('cuda'), test_data_loader) for received_model in received_models]
        # print(sub_accuracies)
        accuracy = test_model(global_model.parameters.to('cuda'), test_data_loader)
        print(f'Accuracy on the test set: {accuracy:.2f}%')
    return global_model


# def nicekd(clients, clients_per_round, total_epochs, local_epochs, difference=True):
#     global_model = init_global_model()
#     client_models = [create_client_model(client.width_factor) for client in clients]
#
#     # create server data
#     server_data = []
#     for subset in [client.data for client in clients]:
#         subset_length = len(subset)
#         one_tenth = subset_length // 10
#         new_indices = subset.indices[:one_tenth]
#         new_subset = Subset(subset.dataset, new_indices)
#         server_data.append(new_subset)
#     # end create server data
#
#     for total_epoch in range(total_epochs):
#         clients_selected, idx = random_choice(clients, clients_per_round, difference=difference)
#         models_selected = [client_models[i] for i in idx]
#
#         received_models = []
#         for client, model in zip(clients_selected, models_selected):
#             model.parameters.to('cuda')
#             client_data_loader = torch.utils.data.DataLoader(client.data, batch_size=64, shuffle=True)
#             optimizer = torch.optim.Adam(model.parameters.parameters(), lr=0.001)
#             criterion = torch.nn.CrossEntropyLoss()
#             for local_epoch in range(local_epochs):
#                 for batch_data, batch_labels in client_data_loader:
#                     model.parameters.train()  # train mode
#                     batch_data, batch_labels = batch_data.to('cuda'), batch_labels.to('cuda')
#                     optimizer.zero_grad()
#                     predictions = model.parameters(batch_data)
#                     loss = criterion(predictions, batch_labels)
#                     loss.backward()
#                     optimizer.step()
#             received_models.append(model)
#         global_model = aggregate(received_models, global_model, server_data, 3)
#
#         # with torch.no_grad():
#         #     global_state_dict = global_model.parameters.state_dict()
#         #     for key in global_state_dict:
#         #         global_state_dict[key] = torch.stack([received_models[i].parameters.state_dict()[key] for i in range(len(received_models))], dim=0).mean(0)
#         #     global_model.parameters.load_state_dict(global_state_dict)
#
#         distributed_models = distribute(global_model, models_selected, server_data, 3)
#         for m, i in zip(distributed_models, idx):
#             client_models[i] = m
#
#         print(f"Epoch {total_epoch + 1}/{total_epochs} completed")
#         test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
#         sub_accuracies = [test_model(received_model.parameters.to('cuda'), test_data_loader) for received_model in received_models]
#         print(sub_accuracies)
#         accuracy = test_model(global_model.parameters.to('cuda'), test_data_loader)
#         print(f'Accuracy on the test set: {accuracy:.2f}%')
#     return global_model


# In[1] ideal_iid
if __name__ == '__main__':
    setup_seed(42)
    cfg = {
        "dataset": "mnist",
        "num_clients": 100,
        "selected_rate": 0.1,
        "total_epoch": 50,
        "local_epoch": 1,
        "difference": True,
        "split_method": "iid",
    }
    # args = get_arguments()
    # cfg = {
    #     "dataset": args.dataset,
    #     "num_clients": args.num_clients,
    #     "selected_rate": args.selected_rate,
    #     "total_epoch": args.total_epoch,
    #     "local_epoch": args.local_epoch,
    #     "difference": args.difference,
    #     "split_method": args.split_method
    # }

    # prepare Net and dataset
    if cfg["dataset"] == "mnist":
        from models import MnistCNN as Net
        train_set, test_set = load_dataset(path="./data", name="mnist")
    elif cfg["dataset"] == "cifar10":
        from models import CifarCNN as Net
        train_set, test_set = load_dataset(path="./data", name="cifar10")
    else:
        raise ValueError("Invalid dataset name")

    clients = create_clients(cfg, train_set)

    # starting federated learning
    num_clients, selected_rate, total_epoch, local_epoch, difference = cfg['num_clients'], cfg["selected_rate"], cfg["total_epoch"], cfg["local_epoch"], cfg["difference"]
    final_global_model = nicekd(clients=clients, clients_per_round=int(selected_rate * num_clients), total_epochs=total_epoch, local_epochs=local_epoch, difference=difference)

    # test final_global_model
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    accuracy = test_model(final_global_model.parameters.to('cuda'), test_data_loader)
    print('Training completed')
    print(f'Accuracy on the test set: {accuracy:.2f}%')

    # test small_model
    server_data = []
    for subset in [client.data for client in clients]:
        subset_length = len(subset)
        one_tenth = subset_length // 10
        new_indices = subset.indices[:one_tenth]
        new_subset = Subset(subset.dataset, new_indices)
        server_data.append(new_subset)
    client_models = [create_client_model(width_factor) for width_factor in Factor.width_factors]
    client_models = distribute(final_global_model, client_models, server_data, 5)
    print([test_model(model.parameters.to('cuda'), test_data_loader) for model in client_models])
