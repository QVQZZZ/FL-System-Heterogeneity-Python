import argparse


def get_arguments():
    """
    读取命令行参数，返回一个argparse.Namespace对象
    :return: argparse.Namespace对象
    """
    parser = argparse.ArgumentParser(description='Federated Learning Configuration')
    parser.add_argument('--dataset', choices=['mnist', 'cifar10'], default='mnist', help='Choose dataset: mnist or cifar10')
    parser.add_argument('--num_clients', type=int, default=100, help='Number of clients participating in federated learning')
    parser.add_argument('--selected_rate', type=float, default=0.1, help='Rate of selected clients in each communication round')
    parser.add_argument('--total_epoch', type=int, default=50, help='Total number of training epochs')
    parser.add_argument('--local_epoch', type=int, default=1, help='Number of local training epochs at each client')
    parser.add_argument('--difference', type=bool, default=False, help='Control client selection: True for selecting all types, False for random selection')
    parser.add_argument('--split_method', choices=['iid', 'dirichlet', 'ideal_iid', 'ideal_dirichlet', 'exclusive_iid', 'exclusive_dirichlet', 'test_exclusive', 'test_small_exp', 'test_small_control'], default='iid', help='Control data splitting and client selection methods')
    return parser.parse_args()
