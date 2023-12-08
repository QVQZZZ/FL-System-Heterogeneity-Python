import numpy as np
from torch.utils.data import random_split, Subset

from fl import Factor, Client
from tool import dirichlet_split_noniid


def create_clients(cfg, train_set):
    """
    若需要新增方法,需要修改的地方为三处: 下面的if-else语句, config_template.yaml, 以及get_args.py.
    """
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
    elif cfg["split_method"] == "ideal_iid":
        # 所有客户端都是强客户端, 重新定义client_width_factors为全1客户端
        client_width_factors = np.random.choice([1], num_clients)
        client_data = random_split(train_set, [len(train_set) // num_clients] * num_clients)
        clients = [Client(width_factor=width_factor, data=data) for width_factor, data in zip(client_width_factors, client_data)]
    elif cfg["split_method"] == "ideal_dirichlet":
        # 所有客户端都是强客户端, 重新定义client_width_factors为全1客户端
        client_width_factors = np.random.choice([1], num_clients)
        client_index = dirichlet_split_noniid(np.array(train_set.targets), alpha=1, n_clients=num_clients)
        client_data = [Subset(train_set, indices) for indices in client_index]
        clients = [Client(width_factor=width_factor, data=data) for width_factor, data in zip(client_width_factors, client_data)]
    elif cfg["split_method"] == "exclusive_iid":
        # split data into different clients (iid)
        # 随后只保留最大的客户端
        client_data = random_split(train_set, [len(train_set) // num_clients] * num_clients)
        clients = [Client(width_factor=width_factor, data=data) for width_factor, data in zip(client_width_factors, client_data)]
        clients = [_ for _ in clients if _.width_factor == max(Factor.width_factors)]
    elif cfg["split_method"] == "exclusive_dirichlet":
        # split data into different clients (dirichlet noniid)
        # 随后只保留最大的客户端
        client_index = dirichlet_split_noniid(np.array(train_set.targets), alpha=1, n_clients=num_clients)
        client_data = [Subset(train_set, indices) for indices in client_index]
        clients = [Client(width_factor=width_factor, data=data) for width_factor, data in zip(client_width_factors, client_data)]
        clients = [_ for _ in clients if _.width_factor == max(Factor.width_factors)]
    elif cfg["split_method"] == "test_exclusive":
        # 将数据0-5类划分到其他客户端上,6-9划分到大客户端,都采用狄利克雷分布(noniid)
        big_client = np.where(client_width_factors == max(Factor.width_factors))[0]
        other_client = np.where(client_width_factors != max(Factor.width_factors))[0]
        dataset_part1 = Subset(train_set, [i for i in range(len(train_set)) if train_set.targets[i] in [6, 7, 8, 9]])
        dataset_part2 = Subset(train_set, [i for i in range(len(train_set)) if train_set.targets[i] not in [6, 7, 8, 9]])
        dataset_part1_targets = [train_set[i][1] for i in dataset_part1.indices]  # Subset对象没有targets属性, 需要手动创建
        dataset_part2_targets = [train_set[i][1] for i in dataset_part2.indices]  # Subset对象没有targets属性, 需要手动创建
        big_client_index = dirichlet_split_noniid(np.array(dataset_part1_targets), alpha=1, n_clients=len(big_client), subset_indices=np.array(dataset_part1.indices))
        big_client_data = [Subset(train_set, indices) for indices in big_client_index]
        other_client_index = dirichlet_split_noniid(np.array(dataset_part2_targets), alpha=1, n_clients=len(other_client), subset_indices=np.array(dataset_part2.indices))
        other_client_data = [Subset(train_set, indices) for indices in other_client_index]
        client_data = big_client_data + other_client_data
        clients = [Client(width_factor=width_factor, data=data) for width_factor, data in zip(client_width_factors, client_data)]
        clients = [_ for _ in clients if _.width_factor == max(Factor.width_factors)]
    elif cfg["split_method"] == "test_small_exp":
        # split data into different clients (test small model)
        # 将0,1,2的样本dirichlet划分到小客户端上, 将剩余的样本dirichlet划分到其他客户端上
        small_client = np.where(client_width_factors == min(Factor.width_factors))[0]
        other_client = np.where(client_width_factors != min(Factor.width_factors))[0]
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
        small_client = np.where(client_width_factors == min(Factor.width_factors))[0]
        other_client = np.where(client_width_factors != min(Factor.width_factors))[0]
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
    return clients
