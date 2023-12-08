import copy
import random
import numpy as np
import torch
from torch.utils.data import random_split, Subset

from datasets import load_dataset
from metrics import test_model
from tool import setup_seed, dirichlet_split_noniid
from fl import Factor, Client, Model


def random_choice(clients, clients_per_round):
    shuffled_clients = random.sample(clients, len(clients))
    selected_clients = shuffled_clients[:clients_per_round]
    return selected_clients


def create_client_model(width_factor):
    parameters = Net(num_classes=10, width_factor=width_factor).cuda()
    client_model = Model(width_factor=width_factor, parameters=parameters)
    return client_model


def get_parameters_from_server(client_model, global_model):
    client_model.parameters.load_state_dict(global_model.parameters.state_dict())


def aggregate(received_models):
    aggregated_model = copy.deepcopy(received_models[0])
    total_params = {name: torch.zeros_like(param.data) for name, param in aggregated_model.parameters.named_parameters()}
    for model in received_models:
        model_params = model.parameters.state_dict()
        for name, param in model_params.items():
            total_params[name] += param
    averaged_params = {name: param / len(received_models) for name, param in total_params.items()}
    aggregated_model.parameters.load_state_dict(averaged_params)
    return aggregated_model


def all_small_fedavg(clients, clients_per_round, total_epochs, local_epochs):
    global_model = create_client_model(min(Factor.width_factors))
    for total_epoch in range(total_epochs):
        clients_selected = random_choice(clients, clients_per_round)
        received_models = []
        for client in clients_selected:
            client_model = create_client_model(client.width_factor)
            get_parameters_from_server(client_model, global_model)
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


if __name__ == '__main__':
    setup_seed(42)
    cfg = {
        "dataset": "mnist",
        "num_clients": 100,
        "selected_rate": 0.1,
        "total_epoch": 50,
        "local_epoch": 1,
        "split_method": "iid",  # iid or dirichlet
    }

    if cfg["dataset"] == "mnist":
        from models import MnistCNN as Net
        train_set, test_set = load_dataset(path="./data", name="mnist")
    elif cfg["dataset"] == "cifar10":
        from models import CifarCNN as Net
        train_set, test_set = load_dataset(path="./data", name="cifar10")
    else:
        raise ValueError("Invalid dataset name")

    # all_small意味着所有客户端都采用最小的模型,但不意味着所有客户端都是小客户端
    # 只是在编程实现中我们将所有客户端都设置为了小客户端而已
    num_clients = cfg["num_clients"]
    client_width_factors = np.random.choice([min(Factor.width_factors)], num_clients)
    if cfg["split_method"] == "iid":
        client_data = random_split(train_set, [len(train_set) // num_clients] * num_clients)
        clients = [Client(width_factor=width_factor, data=data) for width_factor, data in zip(client_width_factors, client_data)]
    elif cfg["split_method"] == "dirichlet":
        client_index = dirichlet_split_noniid(np.array(train_set.targets), alpha=1, n_clients=num_clients)
        client_data = [Subset(train_set, indices) for indices in client_index]
        clients = [Client(width_factor=width_factor, data=data) for width_factor, data in zip(client_width_factors, client_data)]
    else:
        raise ValueError("Invalid split method type")

    num_clients, selected_rate, total_epoch, local_epoch = cfg['num_clients'], cfg["selected_rate"], cfg["total_epoch"], cfg["local_epoch"]
    final_global_model = all_small_fedavg(clients=clients, clients_per_round=int(selected_rate*num_clients), total_epochs=total_epoch, local_epochs=local_epoch)
