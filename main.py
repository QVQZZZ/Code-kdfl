# In[import]
import logging

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader

from dataloaders import CustomDataset, gen_server_dataloader
from lenet import LeNet
from utils import train, test, kd2p, aggregate

# In[global vars and dataset]
client_num = 4
client_keep_ratios = ['1/4', '2/4', '3/4', '1']
dataset_name = "mnist"
partitioner = IidPartitioner(client_num)
fds = FederatedDataset(dataset=dataset_name, partitioners={"train": partitioner})
# partitioner = DirichletPartitioner(num_partitions=client_num, partition_by="label", alpha=0.5, min_partition_size=100, self_balancing=True)

train_loaders = []
train_examples = []
for cid in range(client_num):
    train_set = fds.load_partition(partition_id=cid, split="train")
    train_partition = CustomDataset(train_set, name=dataset_name, split="train")
    train_loader = DataLoader(train_partition, batch_size=64, shuffle=True)
    train_loaders.append(train_loader)
    train_examples.append(len(train_set))

test_set = fds.load_split(split="test")
test_partition = CustomDataset(test_set, name=dataset_name, split="test")
test_loader = DataLoader(test_partition, batch_size=64, shuffle=False)

server_loader = gen_server_dataloader(dataset_name=dataset_name)

# In[main func]
global_epochs = 2
local_epochs = 1
kd_epochs = 1

if __name__ == '__main__':
# def kdfl(debug=False):
    debug = True
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    # Init server model
    global_model = LeNet(p='1').to('cuda')
    if debug:
        loss, acc = test(global_model, test_loader)
        logging.log(logging.DEBUG, f'Init successfully, loss={loss}, acc={acc}')
    # Fed begin
    for global_epoch in range(global_epochs):
        if debug:
            logging.log(logging.DEBUG, '\n' + '='*20 + f' Rounds {global_epoch} starts ' + '='*20)

        # random choice clients (return selected_cids, default all cids: [cid for cid in range(client_num)])
        selected_cids = [cid for cid in range(client_num)]
        selected_train_loaders = [train_loaders[cid] for cid in selected_cids]
        selected_train_examples = [train_examples[cid] for cid in selected_cids]
        if debug:
            logging.log(logging.DEBUG, f'Selected_cids={selected_cids}, examples={selected_train_examples}')

        # distribute models to clients
        selected_client_nets = []
        for cid in selected_cids:
            client_keep_ratio = client_keep_ratios[cid]
            client_net = kd2p(global_model, client_keep_ratio, server_loader, kd_epochs=kd_epochs, temperature=1)
            selected_client_nets.append(client_net)
        if debug:
            for net in selected_client_nets:
                loss, acc = test(net, test_loader)
                logging.log(logging.DEBUG, f'Distribute kd successfully, loss={loss}, acc={acc}')

        # local train
        for net, dataloader in zip(selected_client_nets, selected_train_loaders):
            train(net, dataloader, local_epochs)
        if debug:
            for net in selected_client_nets:
                loss, acc = test(net, test_loader)
                logging.log(logging.DEBUG, f'Local train successfully, loss={loss}, acc={acc}')

        # aggregate models to server
        agg_nets = []
        for net in selected_client_nets:
            agg_net = kd2p(net, '1', server_loader, kd_epochs=kd_epochs, temperature=1,
                           l2=True, last_model=global_model, lambda_reg=0.1)
            agg_nets.append(agg_net)
        if debug:
            for net in agg_nets:
                loss, acc = test(net, test_loader)
                logging.log(logging.DEBUG, f'Aggregate kd successfully, loss={loss}, acc={acc}')

        aggregated_params = aggregate(agg_nets, selected_train_examples)
        global_model.load_state_dict(aggregated_params)
        if debug:
            loss, acc = test(global_model, test_loader)
            logging.log(logging.DEBUG, f'Aggregate successfully, loss={loss}, acc={acc}')
    # eval
    loss, acc = test(global_model, test_loader)
    # return loss, acc