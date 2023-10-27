from torchvision import datasets, transforms


def load_dataset(path, name):
    """
    Load a dataset with the given name.

    Args:
        path (str): The root directory where the dataset will be stored.
        name (str): The name of the dataset ("mnist" or "cifar10").

    Returns:
        train_set: The training dataset.
        test_set: The testing dataset.
    """
    if name == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(root=path, train=True, download=True, transform=transform)  # len == 60000
        test_set = datasets.MNIST(root=path, train=False, download=True, transform=transform)  # len == 10000
    elif name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = datasets.CIFAR10(path, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(path, train=False, download=True, transform=transform_test)
    else:
        raise ValueError("Unsupported dataset name. Supported names are 'mnist' and 'cifar10'.")
    return train_set, test_set

