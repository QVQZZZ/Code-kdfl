"""
glossary
    fkd: Forward Knowledge Distillation
    rkd: Reverse Knowledge Distillation
    aga: intrA-Group Aggregation
    ega: intEr-Group Aggregation
    lt: Local Train
"""


# In[import]
import copy
import logging
import random
from collections import defaultdict

from torch.utils.data import DataLoader

from newdataloaders import load_dataset, dirichlet_distribution, iid_distribution, gen_server_dataloader

from lenet import LeNet
from resnet import ResNet18
from utils import train, test, kd2p, aggregate

# In[global vars and dataset]

client_num = 100
unique_ratios = ['1/4', '2/4', '3/4', '1']
client_ratios = [ratio for ratio in unique_ratios for _ in range(25)]
# client_ratios = ['1/4', '1/4', '2/4', '2/4', '3/4', '3/4', '1', '1']
dataset_name = "cifar10"
batch_size = 256

train_dataset, test_dataset = load_dataset(dataset_name)
# 使用 Dirichlet 分布生成训练数据加载器
train_loaders, train_examples = dirichlet_distribution(train_dataset, client_num, alpha=10000, batch_size=batch_size)
# 或者使用 IID 分布生成训练数据加载器
train_loaders, train_examples = iid_distribution(train_dataset, client_num, batch_size=batch_size)
# 创建测试数据加载器
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# 生成服务器数据加载器
server_loader = gen_server_dataloader(dataset_name=dataset_name, batch_size=256)

# In[main func]
global_epochs = 10
local_epochs = 1
kd_epochs = 1
temperature = 1
lambda_reg = 0.1
debug = True
logger = logging.getLogger('fl_cifar')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

# init
global_model = ResNet18(p='1').to('cuda')
ratio2model4kd = {ratio: ResNet18(p=ratio).to('cuda') for ratio in unique_ratios}  # 该 hashmap 中存放的是用作蒸馏起点的 model
if debug:
    loss, acc = test(global_model, test_loader)
    logger.debug(f'Init successfully, loss={loss}, acc={acc}')
    loss_li, acc_li = zip(*[test(model, test_loader) for _, model in ratio2model4kd.items()])
    logger.debug(f'Ratio2model4kd, loss_li={loss_li}, acc_li={acc_li}')

# Federated starts
for global_epoch in range(global_epochs):
    # 由 global_model 蒸馏出几种 model 并存入 hashmap，该 hashmap 不同于 ratio2model，该 hashmap 中存放的是即将用于 local_train 的 model
    ratio2model4lt = {ratio: kd2p(global_model, ratio2model4kd[ratio], server_loader, kd_epochs=1, temperature=1) for ratio in unique_ratios}
    if debug:
        logger.debug(f'======== EPOCH {global_epoch} ========')
        loss_li, acc_li = zip(*[test(model, test_loader) for _, model in ratio2model4lt.items()])
        logger.debug(f'FKD successfully, loss_li={loss_li}, acc_li={acc_li}')

    # random choice clients (return selected_cids, default all cids: [cid for cid in range(client_num)])
    # selected_cids = [cid for cid in range(client_num)]
    all_samples = list(range(100))
    ranges = [
        list(range(0, 25)),
        list(range(25, 50)),
        list(range(50, 75)),
        list(range(75, 100))
    ]
    selected_cids = [random.choice(r) for r in ranges]
    remaining_samples = [sample for sample in all_samples if sample not in selected_cids]
    selected_from_remaining = random.sample(remaining_samples, 6)
    selected_cids.extend(selected_from_remaining)
    selected_cids.sort()
    logger.debug("Selected samples:", selected_cids)

    # local train
    ratio2models4aga = defaultdict(list)  # 该 hashmap 中的每个 key ratio 对应一个 model 列表，存放相同结构的模型，用于聚合
    for selected_cid in selected_cids:
        train_loader = train_loaders[selected_cid]

        client_ratio = client_ratios[selected_cid]
        net = copy.deepcopy(ratio2model4lt[client_ratio])
        train(net, train_loader, local_epochs)

        ratio2models4aga[client_ratio].append(net)

        if debug:
            loss, acc = test(net, test_loader)
            logger.debug(f'LT successfully, loss={loss}, acc={acc}')

    example4ega = []
    for ratio, model4aga in ratio2models4aga.items():
        _ = [client_ratios[i] if i in selected_cids else '0' for i in range(client_num)]  # mask not selected clients
        client_indices = [idx for idx, r in enumerate(_) if r == ratio]
        example4aga = [train_examples[idx] for idx in client_indices]
        aggregated_params = aggregate(model4aga, example4aga)
        ratio2model4kd[ratio].load_state_dict(aggregated_params)
        example4ega.append(sum(example4aga))
    if debug:
        loss_li, acc_li = zip(*[test(model, test_loader) for _, model in ratio2model4kd.items()])
        logger.debug(f'AGA successfully, loss_li={loss_li}, acc_li={acc_li}')

    l2_model = ratio2model4kd['1']
    model4ega = [kd2p(model, global_model, server_loader, kd_epochs=1, temperature=1,
                      l2=True, last_model=l2_model, lambda_reg=0.1) if ratio != '1' else model
                 for ratio, model in ratio2model4kd.items()]
    if debug:
        loss_li, acc_li = zip(*[test(model, test_loader) for model in model4ega])
        logger.debug(f'RKD successfully, loss_li={loss_li}, acc_li={acc_li}')
    aggregated_params = aggregate(model4ega, example4ega)
    global_model.load_state_dict(aggregated_params)
    # loss, acc = test(global_model, test_loader)
    # print(global_epoch, loss, acc)

    loss, acc = test(global_model, test_loader)
    if debug:
        logger.debug(f'EGA successfully, loss={loss}, acc={acc}')
    else:
        print(loss, acc)
loss, acc = test(global_model, test_loader)
print(loss, acc)
