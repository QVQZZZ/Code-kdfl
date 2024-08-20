# In[Glossary]
"""
glossary
    fkd: Forward Knowledge Distillation
    rkd: Reverse Knowledge Distillation
    aga: intrA-Group Aggregation
    ega: intEr-Group Aggregation
    lt: Local Train
"""

# In[Import]
import argparse
import copy
import logging
import random
from collections import defaultdict

from torch.utils.data import DataLoader

from newdataloaders import load_dataset, dirichlet_distribution, iid_distribution, gen_server_dataloader
from lenet import LeNet
from resnet import ResNet18
from utils import train, test, aggregate
from utils_kd import kd2p, LossStrategy

# In[Configuration and setup]
client_num = 8
unique_ratios = ['1/4', '2/4', '3/4', '1']
client_ratios = ['1/4', '1/4', '2/4', '2/4', '3/4', '3/4', '1', '1']
dataset_name = "mnist"
network = "lenet"
global_epochs = 1
local_epochs = 1
batch_size = 256
fkd_epochs = 1
rkd_epochs = 1
temperature_fkd = 1
temperature_rkd = 1
lambda_reg = 0.1  # or increase while federated learning?
iid = True
alpha = 0.5
debug = True

# In[Initialize logging]
logger = logging.getLogger('fl')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

# In[Prepare datasets]
train_dataset, test_dataset = load_dataset(dataset_name)
if iid:
    train_loaders, train_examples = iid_distribution(train_dataset, client_num, batch_size=batch_size)
else:
    train_loaders, train_examples = dirichlet_distribution(train_dataset, client_num, alpha=alpha, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
server_loader = gen_server_dataloader(dataset_name=dataset_name, batch_size=batch_size)

# In[Main function]
# Init global model and 'small' models
# 该 hashmap 中存放的是用作蒸馏起点的 model
clz = ResNet18 if network == "resnet18" else LeNet
ratio2model4kd = {ratio: clz(p=ratio).to('cuda') for ratio in unique_ratios}
global_model = clz(p='1').to('cuda')

# Federated starts
for global_epoch in range(global_epochs):
    print('='*20, f'EPOCH {global_epoch}', '='*20)

    # 1. FKD: forward knowledge distillation
    # 由 global_model 蒸馏出几种 model 并存入 hashmap, 该 hashmap 中存放的是即将用于 local_train 的 model
    ratio2model4lt = {ratio: kd2p(global_model, ratio2model4kd[ratio],
                                  server_loader, kd_epochs=fkd_epochs, temperature=temperature_fkd,
                                  loss_strategy=LossStrategy.SOFT_ONLY, alpha=None,
                                  l2=False, target_model=None, lambda_reg=None,
                                  verbose=False)
                      for ratio in unique_ratios}
    if debug:
        loss_li, acc_li = zip(*[test(model, test_loader) for _, model in ratio2model4lt.items()])
        print(f'FKD successfully, loss_li={loss_li}, acc_li={acc_li}')

    # 1.1 Random choice clients (return selected_cids, default all cids: [cid for cid in range(client_num)])
    selected_cids = [cid for cid in range(client_num)]
    if debug:
        pass

    # 2. LT: local train
    # 该 hashmap 中的每个 key ratio 对应一个 model 列表, 存放相同结构的模型, 用于聚合
    ratio2models4aga = defaultdict(list)
    for selected_cid in selected_cids:
        train_loader = train_loaders[selected_cid]
        train_example = train_examples[selected_cid]
        # 根据选择的客户端来分配某个 ratio 的网络并进行训练
        client_ratio = client_ratios[selected_cid]
        net = copy.deepcopy(ratio2model4lt[client_ratio])
        train(net, train_loader, local_epochs)
        ratio2models4aga[client_ratio].append(net)
    if debug:
        loss_li, acc_li = zip(*[test(model, test_loader) for model in sum([*ratio2models4aga.values()], [])])
        print(f'LT successfully, loss_li={loss_li}, acc_li={acc_li}')

    # 3. AGA: intra-group aggregation
    example4ega = []
    for ratio, model4aga in ratio2models4aga.items():
        client_indices = [idx for idx, r in enumerate(client_ratios) if r == ratio]
        example4aga = [train_examples[idx] for idx in client_indices]
        aggregated_params = aggregate(model4aga, example4aga)
        ratio2model4kd[ratio].load_state_dict(aggregated_params)
        example4ega.append(sum(example4aga))
    if debug:
        loss_li, acc_li = zip(*[test(model, test_loader) for _, model in ratio2model4kd.items()])
        print(f'AGA successfully, loss_li={loss_li}, acc_li={acc_li}')

    # 4. RKD: reverse knowledge distillation
    l2_model = ratio2model4kd['1']

    model4ega = [kd2p(model, global_model,
                      server_loader, kd_epochs=rkd_epochs, temperature=temperature_rkd,
                      loss_strategy=LossStrategy.SOFT_ONLY, alpha=None,
                      l2=True, target_model=l2_model, lambda_reg=lambda_reg)
                 if ratio != '1' else model
                 for ratio, model in ratio2model4kd.items()]
    if debug:
        loss_li, acc_li = zip(*[test(model, test_loader) for model in model4ega])
        print(f'RKD successfully, loss_li={loss_li}, acc_li={acc_li}')

    # 5. EGA: inter-group aggregation (default debug)
    aggregated_params = aggregate(model4ega, example4ega)
    global_model.load_state_dict(aggregated_params)
    loss, acc = test(global_model, test_loader)
    print(f'EGA successfully: loss={loss}, acc={acc}')

# In[Training ends and post-processing]
loss, acc = test(global_model, test_loader)
loss_li, acc_li = zip(*[test(model, test_loader) for _, model in ratio2model4kd.items()])
ratio2model4lt = {ratio: kd2p(global_model, ratio2model4kd[ratio],
                              server_loader, kd_epochs=fkd_epochs, temperature=temperature_fkd,
                              loss_strategy=LossStrategy.SOFT_ONLY, alpha=None,
                              l2=False, target_model=None, lambda_reg=None,
                              verbose=False)
                  for ratio in unique_ratios}
loss_li_kd, acc_li_kd = zip(*[test(model, test_loader) for _, model in ratio2model4lt.items()])
print(f'Federated learning successfully\n'
      f'Global_model: loss={loss}, acc={acc}\n'
      f'Client_models: loss_li={loss_li}, acc_li={acc_li}\n'
      f'Client_models: loss_li={loss_li_kd}, acc_li={acc_li_kd}')
