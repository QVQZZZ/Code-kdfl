import copy
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class LossStrategy:
    COMBINATION = 0  # 采用 soft loss 和 hard loss 的线性组合
    MIXTURE = 1      # 采用混合的 label, 以 soft loss 的方式计算 loss total
    SOFT_ONLY = 2    # 适用于无标签数据集, 不需要 labels

    STRATEGY_TO_STR = {
        COMBINATION: "Combination",
        MIXTURE: "Mixture",
        SOFT_ONLY: "SoftOnly"
    }

    @staticmethod
    def to_string(strategy):
        return LossStrategy.STRATEGY_TO_STR.get(strategy, f"Unknown Strategy, "
                                                          f"should be one of {LossStrategy.STRATEGY_TO_STR.keys()}, "
                                                          f"check the STRATEGY_TO_STR dictionary.")

class SoftLoss(nn.Module):
    def __init__(self, temperature):
        super(SoftLoss, self).__init__()
        self.temperature = temperature
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits):
        """
        计算 Soft Loss (KL 散度)
        Args:
            student_logits: 学生模型的预测输出 (未经过 softmax)
            teacher_logits: 教师模型的预测输出 (未经过 softmax)
        Returns:
            计算得到的损失值 (torch.Tensor)
        """
        student_probs = self.log_softmax(student_logits / self.temperature)
        teacher_probs = self.softmax(teacher_logits / self.temperature)
        loss_soft = self.kl_div_loss(student_probs, teacher_probs)
        return loss_soft


class HardLoss(nn.Module):
    def __init__(self):
        super(HardLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        计算Hard Loss (交叉熵)
        Args:
            logits: 模型的预测输出 (未经过 softmax, nn.CrossEntropyLoss 会自动进行 softmax)
            labels: 真实标签 (标量, 不需要 one-hot 编码, nn.CrossEntropyLoss 会自动进行 one-hot 编码)
        Returns:
            计算得到的损失值 (torch.Tensor)
        """
        loss_hard = self.cross_entropy_loss(logits, labels)
        return loss_hard


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, model, target_model):
        """
        计算L2 Loss (二范数)
        Args:
            model (nn.Module): 当前模型
            target_model (nn.Module): 目标模型, 用于计算差异
        Returns:
            计算得到的损失值 (torch.Tensor)
        """
        loss_l2 = 0.0
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            loss_l2 += torch.norm(param - target_param, p=2)
        return loss_l2


class KnowledgeDistillationLossWithRegularization(nn.Module):
    """
    带正则损失和知识蒸馏损失的函数类, 该类支持三种策略 (COMBINATION, MIXTURE, SOFT_ONLY) 以及可选的L2正则化
    Args:
        temperature (float): 蒸馏温度, 用于调整 softmax 的平滑度
        l2 (bool): 是否启用 L2 正则化
        lambda_reg (float): L2 正则化系数, 仅在 l2=True 时需要
        loss_strategy (int): 损失计算策略, 支持 COMBINATION, MIXTURE, SOFT_ONLY
        alpha (float): 在 COMBINATION 策略下, soft loss 和 hard loss 的权重系数, 仅在 COMBINATION 时需要
    Raises:
        ValueError: 如果 loss_strategy 和相应参数不匹配, 或 L2 正则化需要的参数未提供时, 会抛出异常
    """
    def __init__(self, temperature,      # 蒸馏参数
                 l2, lambda_reg,         # L2 正则化参数
                 loss_strategy, alpha):  # 损失策略参数
        super(KnowledgeDistillationLossWithRegularization, self).__init__()
        # 是否采用 L2 正则化, 若 L2 = True 则需要设置 lambda_reg
        self.l2 = l2
        self.lambda_reg = lambda_reg  # 仅用于 L2 = True, 若 L2 = True 则需要设置 lambda_reg
        if self.l2 and self.lambda_reg is None:
            raise ValueError("L2 regularization requires a valid lambda_reg.")

        # 采用何种损失策略, 若 loss_strategy 为 COMBINATION(=0) 则需要设置 alpha
        self.loss_strategy = loss_strategy
        self.alpha = alpha  # 仅用于 COMBINATION 策略, 若 loss_strategy 为 COMBINATION 则需要设置 alpha
        if self.loss_strategy == LossStrategy.COMBINATION and self.alpha is None:
            raise ValueError("Combination Loss strategy requires a valid alpha.")

        self.soft_loss = SoftLoss(temperature)
        self.hard_loss = HardLoss()
        self.l2_loss = L2Loss()

    def forward(self, student_logits, teacher_logits, labels, model, target_model=None, verbose=False):
        """
        计算损失值, 根据初始化时选择的策略和参数, 计算最终的总损失值
        Args:
            student_logits (torch.Tensor): 学生模型的预测输出 (未经过 softmax)
            teacher_logits (torch.Tensor): 教师模型的预测输出 (未经过 softmax)
            labels (torch.Tensor): 真实标签, 在 COMBINATION 和 MIXTURE 策略下使用
            model (nn.Module): 当前学生模型, 用于 L2 正则化
            target_model (nn.Module, optional): 目标模型, 用于计算 L2 正则化损失
            verbose (bool, optional): 如果为 True, 则输出详细的损失信息
        Returns:
            torch.Tensor: 计算得到的总损失值
        Raises:
            ValueError: 如果参数配置不正确或缺失, 抛出异常
        """
        if self.l2 and target_model is None:
            raise ValueError("L2 regularization requires a target model to compare against.")
        if self.loss_strategy in [LossStrategy.COMBINATION, LossStrategy.MIXTURE] and labels is None:
            raise ValueError("Combination / Mixture Loss strategy require labels.")

        loss_total = 0.0  # 总损失, 无论采用何种 Strategy 以及是否启用 L2 正则化, 都需要计算, 必定存在
        loss_soft, loss_hard, loss_combination, loss_mixture, loss_l2 = None, None, None, None, None  # 是否存在取决于 Strategy 和 L2

        if self.loss_strategy == LossStrategy.COMBINATION:
            # 计算 soft loss 和 hard loss 的线性组合, 然后计算 loss total
            loss_soft = self.soft_loss(student_logits, teacher_logits)
            loss_hard = self.hard_loss(student_logits, labels)
            loss_combination = self.alpha * loss_hard + (1 - self.alpha) * loss_soft
            loss_total += loss_combination

        elif self.loss_strategy == LossStrategy.MIXTURE:
            # 将 soft label 和 hard label 混合, 然后用混合的 label, 以 soft loss 的方式计算 loss total
            labels_one_hot = F.one_hot(labels, num_classes=student_logits.size(1)).float().to(student_logits.device)
            student_probs = self.soft_loss.log_softmax(student_logits / self.soft_loss.temperature)
            teacher_probs = self.soft_loss.softmax(teacher_logits / self.soft_loss.temperature)
            mixed_targets = 0.5 * (teacher_probs + labels_one_hot)
            loss_mixture = self.soft_loss.kl_div_loss(student_probs, mixed_targets)
            loss_total += loss_mixture

        elif self.loss_strategy == LossStrategy.SOFT_ONLY:
            # 仅计算 soft loss, 适用于无标签数据集
            loss_soft = self.soft_loss(student_logits, teacher_logits)
            loss_total += loss_soft

        # 如果启用了 L2 正则化, 则再添加 L2 Loss
        if self.l2:
            loss_l2 = self.l2_loss(model, target_model)
            loss_total += self.lambda_reg * loss_l2

        if verbose:
            loss_strategy_str = LossStrategy.to_string(self.loss_strategy)
            print(f"Strategy: {loss_strategy_str}, Use L2: {self.l2}")
            print(f"Loss (Total): {loss_total:.5g}")

            if self.loss_strategy == LossStrategy.COMBINATION:
                print(f"Loss (Soft): {loss_soft:.5g}, Loss (Hard): {loss_hard:.5g},"
                      f"Loss (Combination): {loss_combination:.5g}, Alpha: {self.alpha:.5g}")
            elif self.loss_strategy == LossStrategy.MIXTURE:
                print(f"Loss (Mixture): {loss_mixture:.5g}")
            elif self.loss_strategy == LossStrategy.SOFT_ONLY:
                print(f"Loss (Soft): {loss_soft:.5g}")

            if self.l2:
                print(f"Loss (L2): {loss_l2:.5g}, Lambda: {self.lambda_reg:.5g}")

        return loss_total


def kd2p(model: nn.Module, dest: Union[str, nn.Module],  # 教师模型, Union[学生模型大小, 学生模型]
         server_data: DataLoader, kd_epochs: int, temperature: float,                 # 蒸馏参数
         loss_strategy: int = LossStrategy.COMBINATION, alpha: float = 0.5,           # 损失策略参数
         l2: bool = False, target_model: nn.Module = None, lambda_reg: float = None,  # L2 正则化参数
         verbose: bool = False) -> nn.Module:
    """
    通过知识蒸馏将教师模型的知识迁移到学生模型中支持使用标签的有监督学习和无标签的无监督学习, 并且可以选择是否进行L2正则化
    Args:
        model (nn.Module): 教师模型, 用于提供知识指导
        dest (Union[str, nn.Module]): 学生模型的结构, 或者直接传入一个已有的学生模型
        server_data (DataLoader): 服务器上的数据, 用于知识蒸馏
        kd_epochs (int): 蒸馏的训练轮数
        temperature (float): 蒸馏温度, 用于调整 softmax 的平滑度
        loss_strategy (int, optional): 蒸馏中损失计算策略, 取决于是否采用无标签数据集和损失的构成, 默认为 COMBINATION
        alpha (float, optional): 在 COMBINATION 策略下, soft loss 和 hard loss 的权重系数, 默认为 0.5
        l2 (bool, optional): 是否启用 L2 正则化, 默认为 False
        target_model (nn.Module, optional): 目标模型, 用于计算 L2 正则化损失, 仅在 l2=True 时需要
        lambda_reg (float, optional): L2 正则化系数, 仅在 l2=True 时需要
        verbose (bool, optional): 是否输出详细的训练过程中的损失信息, 默认为 False
    Returns:
        nn.Module: 蒸馏后的学生模型
    Raises:
        ValueError: 如果 dest 模型未正确配置, 或参数配置不正确时, 抛出异常
    """
    if isinstance(dest, str):
        dest_model = type(model)(p=dest).to('cuda')  # 从随机初始化开始训练
    else:
        dest_model = copy.deepcopy(dest).to('cuda')  # 从现有模型开始训练

    dest_model.train()
    criterion = KnowledgeDistillationLossWithRegularization(temperature=temperature,
                                                            l2=l2, lambda_reg=lambda_reg,
                                                            loss_strategy=loss_strategy, alpha=alpha)
    optimizer = torch.optim.Adam(dest_model.parameters())

    for epoch in range(kd_epochs):
        for inputs, labels in server_data:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            student_logits = dest_model(inputs)
            with torch.no_grad():
                teacher_logits = model(inputs)
            loss = criterion(student_logits, teacher_logits, labels, dest_model, target_model, verbose)
            loss.backward()
            optimizer.step()

    return dest_model
