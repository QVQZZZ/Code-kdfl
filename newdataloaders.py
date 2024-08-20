import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


def dirichlet_distribution(dataset: Dataset, num_clients: int, alpha: float, batch_size: int = 256):
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = np.max(labels) + 1
    client_indices = [[] for _ in range(num_clients)]
    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0]
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        client_indices_split = np.split(class_indices, proportions)
        for i in range(num_clients):
            client_indices[i].extend(client_indices_split[i])
    data_loaders = [DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True) for indices in client_indices]
    data_lengths = [len(indices) for indices in client_indices]
    return data_loaders, data_lengths


def iid_distribution(dataset: Dataset, num_clients: int, batch_size: int = 256):
    all_indices = np.arange(len(dataset))
    np.random.shuffle(all_indices)
    client_indices = np.array_split(all_indices, num_clients)
    data_loaders = [DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True) for indices in client_indices]
    data_lengths = [len(indices) for indices in client_indices]
    return data_loaders, data_lengths


def load_dataset(dataset_name: str):
    transform_dict = {
        "cifar10": {
            "train": transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
            "test": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        },
        "mnist": {
            "train": transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            "test": transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1325,), (0.3105,))
            ])
        },
        "fashion": {
            "train": transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ]),
            "test": transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        }
    }

    if dataset_name not in transform_dict:
        raise ValueError("Unsupported dataset.")

    train_transform = transform_dict[dataset_name]["train"]
    test_transform = transform_dict[dataset_name]["test"]

    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=test_transform)
    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=test_transform)
    elif dataset_name == "fashion":
        train_dataset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=test_transform)
    return train_dataset, test_dataset


def gen_server_dataloader(dataset_name: str, batch_size: int = 256):
    from torchvision import datasets, transforms
    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        server_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        server_dataloader = torch.utils.data.DataLoader(server_dataset, batch_size=batch_size, shuffle=True)
    elif dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        server_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
        server_dataloader = torch.utils.data.DataLoader(server_dataset, batch_size=batch_size, shuffle=True)
    elif dataset_name == "fashion":
        transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
        ])
        server_dataset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
        server_dataloader = torch.utils.data.DataLoader(server_dataset, batch_size=batch_size, shuffle=True)
    return server_dataloader


# 使用示例
def main():
    client_num = 10
    dataset_name = "fashion"  # 或 "cifar10"
    batch_size = 256
    # 加载数据集
    train_dataset, test_dataset = load_dataset(dataset_name)
    # 使用 Dirichlet 分布生成训练数据加载器
    train_loaders, train_examples = dirichlet_distribution(train_dataset, client_num, alpha=10000, batch_size=batch_size)
    # 或者使用 IID 分布生成训练数据加载器
    train_loaders, train_examples = iid_distribution(train_dataset, client_num, batch_size=batch_size)
    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 输出信息
    for i, (loader, examples) in enumerate(zip(train_loaders, train_examples)):
        print(f"Client {i}: {examples} samples")
    print(f"Total test samples: {len(test_dataset)}")


if __name__ == "__main__":
    main()




