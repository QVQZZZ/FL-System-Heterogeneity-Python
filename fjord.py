# In[0]
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from tool import *
from datasets import *
from metrics import *
from fl import *


def init_global_model():
    """
    初始化width_factor=1的全局模型
    :return: 一个初始化后的width_factor=1的模型
    """
    parameters = SimpleCNN(num_classes=10, width_factor=1.0).cuda()
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
        parameters = SimpleCNN(num_classes=10, width_factor=width_factor).cuda()
        client_model = Model(width_factor=width_factor, parameters=parameters)
        return client_model
    else:
        width_factors = np.array(Factor.width_factors)
        width_factors = width_factors[width_factors <= width_factor]
        client_models = []
        for width_factor in width_factors:
            parameters = SimpleCNN(num_classes=10, width_factor=width_factor).cuda()
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
    :return: Client类组成的列表
    """
    if difference:
        # 若为True,则保证选出的客户端包含所有宽度的客户端
        while True:
            shuffled_clients = random.sample(clients, len(clients))
            selected_clients = shuffled_clients[:clients_per_round]
            clients_type = set([client.width_factor for client in selected_clients])
            if len(clients_type) == len(Factor.width_factors):
                return selected_clients
    else:
        # 若为False,也至少要保证选出的客户端中包含一个宽度为1的客户端,避免聚合时出现大量的0
        while True:
            shuffled_clients = random.sample(clients, len(clients))
            selected_clients = shuffled_clients[:clients_per_round]
            for selected_client in selected_clients:
                if selected_client.width_factor == 1:
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

    :undone:如果缺少宽度为1的客户端,那么可能出现参数大量为0,因此在random_choice函数中保证了必须出现宽度为1的客户端
    :undone: 建议新增一个功能,在出现这种情况时让aggregate保留上一轮通信的参数,但是这种方式不适合用函数了,因为给aggregate添加一个全局模型作为参数很不优雅
    :undone: 后续写其他实验时可以考虑用Server类,将aggregate设计为一个方法,将全局模型设置为属性,这样就能很方便地添加这个功能了
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


def fjord(clients, clients_per_round, total_epochs, local_epochs, difference=True):
    global_model = init_global_model()
    for total_epoch in range(total_epochs):
        clients_selected = random_choice(clients, clients_per_round, difference=difference)
        received_models = []
        for client in clients_selected:
            client_models = create_client_model(client.width_factor, multiple=True)
            teacher_model = max(client_models, key=lambda model: model.width_factor)
            student_models = [model for model in client_models if model != teacher_model]

            # 先训练一遍teacher
            teacher_model = get_parameters_from_server(teacher_model, global_model)
            client_data_loader = torch.utils.data.DataLoader(client.data, batch_size=64, shuffle=True)
            optimizer = torch.optim.Adam(teacher_model.parameters.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            for local_epoch in range(local_epochs):
                for batch_data, batch_labels in client_data_loader:
                    teacher_model.parameters.train()
                    batch_data, batch_labels = batch_data.to('cuda'), batch_labels.to('cuda')
                    optimizer.zero_grad()
                    predictions = teacher_model.parameters(batch_data)
                    loss = criterion(predictions, batch_labels)
                    loss.backward()
                    optimizer.step()
            received_models.append(teacher_model)
            # 再训练每个student(采用KD)
            for student_model in student_models:
                student_model = get_parameters_from_server(student_model, global_model)
                client_data_loader = torch.utils.data.DataLoader(client.data, batch_size=64, shuffle=True)
                optimizer = torch.optim.Adam(student_model.parameters.parameters(), lr=0.001)
                kd_criterion = nn.KLDivLoss(reduction='batchmean')
                student_criterion = nn.CrossEntropyLoss()
                for local_epoch in range(local_epochs):
                    for batch_data, batch_labels in client_data_loader:
                        student_model.parameters.train()
                        batch_data, batch_labels = batch_data.to('cuda'), batch_labels.to('cuda')
                        optimizer.zero_grad()
                        with torch.no_grad():
                            predictions_teacher = teacher_model.parameters(batch_data)
                        predictions_student = student_model.parameters(batch_data)
                        student_loss = student_criterion(predictions_student, batch_labels)
                        temperature = 1; alpha = 0.5
                        kd_loss = kd_criterion(torch.log_softmax(predictions_student / temperature, dim=1),
                                               torch.softmax(predictions_teacher / temperature, dim=1))
                        loss = (1 - alpha) * student_loss + alpha * kd_loss
                        loss.backward()
                        optimizer.step()
                received_models.append(student_model)
        global_model = aggregate(received_models)
        print(f"Epoch {total_epoch + 1}/{total_epochs} completed")

        test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
        accuracy = test_model(global_model.parameters.to('cuda'), test_data_loader)
        print(f'Accuracy on the test set: {accuracy:.2f}%')
    return global_model


if __name__ == '__main__':
    setup_seed(100)
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
        from models import MnistCNN as SimpleCNN
        train_set, test_set = load_dataset(path="./data", name="mnist")
    elif cfg["dataset"] == "cifar10":
        from models import CifarCNN as SimpleCNN
        train_set, test_set = load_dataset(path="./data", name="cifar10")
    else:
        raise ValueError("Invalid dataset name")

    clients = create_clients(cfg, train_set)

    # starting federated learning
    num_clients, selected_rate, total_epoch, local_epoch, difference = cfg['num_clients'], cfg["selected_rate"], cfg["total_epoch"], cfg["local_epoch"], cfg["difference"]
    final_global_model = fjord(clients=clients, clients_per_round=int(selected_rate*num_clients), total_epochs=total_epoch, local_epochs=local_epoch, difference=difference)

    # test final_global_model
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    accuracy = test_model(final_global_model.parameters.to('cuda'), test_data_loader)
    print('Training completed')
    print(f'Accuracy on the test set: {accuracy:.2f}%')

    # test sub_models
    sub_models = [create_client_model(width_factor) for width_factor in Factor.width_factors]
    sub_models = [get_parameters_from_server(sub_model, final_global_model) for sub_model in sub_models]
    accuracies = [test_model(sub_model.parameters.to('cuda'), test_data_loader) for sub_model in sub_models]
    for idx, sub_model in enumerate(sub_models):
        print(f'Accuracy for sub_model with width_factor {sub_model.width_factor}: {accuracies[idx]:.2f}%')