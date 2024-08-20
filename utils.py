import copy
from typing import Union

import torch
import torch.nn as nn


DEVICE = torch.device("cuda")


def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images).to(DEVICE)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"LOCAL TRAIN: epoch {epoch + 1}, train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test_monitor set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images).to(DEVICE)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return round(loss, 5), round(accuracy, 4)


class LossStrategy:
    COMBINATION = 0
    MIXTURE = 1

class KnowledgeDistillationLossWithRegularization(nn.Module):
    def __init__(self, temperature, lambda_reg, alpha, loss_strategy):
        super(KnowledgeDistillationLossWithRegularization, self).__init__()
        self.temperature = temperature
        self.lambda_reg = lambda_reg if lambda_reg else 0
        self.alpha = alpha  # only used in COMBINATION
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs, teacher_outputs, l2, model, old_model, verbose=False):
        outputs = self.log_softmax(outputs / self.temperature)
        teacher_outputs = self.softmax(teacher_outputs / self.temperature)
        loss_kd = self.kldiv_loss(outputs, teacher_outputs)

        l2_regularization = 0.0
        if l2:
            for param, old_param in zip(model.parameters(), old_model.parameters()):
                l2_regularization += torch.norm(param - old_param, p=2)
        # 总损失 = 知识蒸馏损失 + L2正则化项
        loss_total = loss_kd + self.lambda_reg * l2_regularization
        if verbose:
            print(f'loss_kd:{loss_kd:<.5g}, loss_l2:{l2_regularization:<.5g}, lambda:{self.lambda_reg:<.5g}, '
                  f'loss_kd/(lambda*loss_l2)={(loss_kd / (self.lambda_reg * l2_regularization)):<.5g}')
        return loss_total


def kd2p(model, dest: Union[str, nn.Module], server_data, kd_epochs, temperature,
         l2=False, last_model=None, lambda_reg=None,
         verbose=False):
    """
    采用知识蒸馏将模型变成指定的keep_ratio大小，若dest为Module，该函数不会改变dest
    :param model: 教师模型
    :param dest: 蒸馏后的学生模型的大小dest_keep_ratio(如'3/4')并从随机初始化开始训练，或直接传入一个现成的学生模型dest_model并在此基础上开始训练
    :param server_data: 服务器上用于蒸馏的数据
    :param kd_epochs: 蒸馏的epoch次数
    :param temperature: 蒸馏温度

    :param l2: 是否采用正则化，如果设置为True，则必须要设置last_model和lambda_reg，否则报错
    :param last_model: 蒸馏回去的时候需要保证与last_model接近，避免模型太过发散
    :param lambda_reg: 正则化强度

    :param verbose: 用于控制是否要输出kd_loss和l2_loss的详细信息
    :return: 蒸馏后的模型
    """
    if isinstance(dest, str):
        dest_model = type(model)(p=dest).to('cuda')
        # from resnet import ResNet18
        # dest_model = ResNet18(p=dest).to('cuda')
    else:
        dest_model = copy.deepcopy(dest).to('cuda')
    dest_model.train()
    criterion = KnowledgeDistillationLossWithRegularization(temperature=temperature, lambda_reg=lambda_reg)
    optimizer = torch.optim.Adam(dest_model.parameters())
    for epoch in range(kd_epochs):
        for inputs, labels in server_data:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = dest_model(inputs)
            with torch.no_grad():
                teacher_outputs = model(inputs)
            loss = criterion(outputs, teacher_outputs, l2, dest_model, last_model, verbose)
            loss.backward()
            optimizer.step()
    return dest_model


def aggregate(agg_nets, selected_train_examples):
    """
    Aggregates the parameters of multiple neural networks using Federated Averaging.

    Parameters:
    - agg_nets: List of nn.Module, the locally trained neural network models.
    - selected_train_examples: List of int, the number of training examples used to train each model.

    Returns:
    - aggregated_params: Dict[str, torch.Tensor], the aggregated parameters of the neural network.
    """

    # Ensure that the number of models and the number of training examples are the same
    assert len(agg_nets) == len(
        selected_train_examples), "The length of agg_nets and selected_train_examples must be the same."

    # Total number of training examples
    total_train_examples = sum(selected_train_examples)

    # Initialize a dictionary to store the sum of the parameters weighted by the number of training examples
    agg_params = {name: torch.zeros_like(param.data) for name, param in agg_nets[0].named_parameters()}

    # Sum the weighted parameters of each model
    for net, num_examples in zip(agg_nets, selected_train_examples):
        for name, param in net.named_parameters():
            agg_params[name] += param.data * (num_examples / total_train_examples)

    return agg_params
