import numpy as np


def dirichlet_split_noniid(train_labels, alpha, n_clients, subset_indices=None):
    """
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    """
    n_classes = len(set(train_labels))
    # n_classes*n_clients 的二维array,每一行代表这个class在各个client上的分布比例
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # n_classes*该类别的样本数 的二维列表,每一行代表这个class对应原始数据集的样本索引
    if subset_indices is not None:
        # 如果提供了子集在原始数据集中的索引位置，需要将它映射回原始数据集
        class_index = [subset_indices[train_labels == y] for y in np.unique(train_labels)]
    else:
        # 否则，使用默认的方式
        class_index = [np.argwhere(train_labels == y).flatten() for y in np.unique(train_labels)]

    # 记录n_clients个client分别对应的样本索引集合
    client_index = [[] for _ in range(n_clients)]
    for class_k_index, distribution in zip(class_index, label_distribution):
        # np.cumsum(distribution)[:-1]表示该类样本在客户端上的累积概率密度,去除最后一项(100%). Array广播乘该类的样本数len(class_k_index)变成了累积分布函数,去除最后一项。
        # np.split后就是这个类别的样本在每个客户端上的索引
        for i, class_k_client_n_index in enumerate(np.split(class_k_index, (np.cumsum(distribution)[:-1] * len(class_k_index)).astype(int))):
            client_index[i] += [class_k_client_n_index]

    client_index = [np.concatenate(_) for _ in client_index]

    return client_index


