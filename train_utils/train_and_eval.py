import math
import torch
from torch.nn import functional as F
import train_utils.distributed_utils as utils
import numpy as np
import src.ssim_Loss
##导入新的库
from sklearn.metrics import accuracy_score, precision_score, recall_score

##添加新的任务评价指标
##通过混淆矩阵来测Acc（准确率），Precision（精确率），Recall（召回率）
ssim_loss = src.ssim_Loss.SSIM(window_size=11,size_average=True)
tensor_1 = [torch.ones(1, device='cuda:0', requires_grad=True) for _ in range(7)]
def criterion(inputs, target):
    losses = [F.binary_cross_entropy_with_logits(inputs[i], target) for i in range(len(inputs))]

    #a=[ssim_loss(inputs[i], target) for i in range(len(inputs))]
    # total_loss_bce = sum(losses)
    # total_loss_ssim = sum(a)
    # tensor = sum(tensor_1)
    # return total_loss_bce+tensor-total_loss_ssim

    loss = sum(losses)
    return loss


def evaluate(model, data_loader, device):
    model.eval()
    mae_metric = utils.MeanAbsoluteError()
    f1_metric = utils.F1Score()
    precision = utils.F1Score_pre()
    recall = utils.F1Score_rec()
    acc_1 = utils.F1Score_acc_1()
    acc_2 = utils.F1Score_acc_2()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images, targets = images.to(device), targets.to(device)
            output = model(images)

            mae_metric.update(output, targets)
            f1_metric.update(output, targets)
            precision.update(output, targets)
            recall.update(output, targets)
            acc_1.update(output, targets)
            acc_2.update(output, targets)


        mae_metric.gather_from_all_processes()
        f1_metric.reduce_from_all_processes()
        precision.reduce_from_all_processes()
        #recall.reduce_from_all_processes()
        acc_1.reduce_from_all_processes()
        acc_2.reduce_from_all_processes()


    return mae_metric, f1_metric ,precision ,acc_2


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-4):
    params_group = [{"params": [], "weight_decay": 0.},  # no decay
                    {"params": [], "weight_decay": weight_decay}]  # with decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            # bn:(weight,bias)  conv2d:(bias)  linear:(bias)
            params_group[0]["params"].append(param)  # no decay
        else:
            params_group[1]["params"].append(param)  # with decay

    return params_group
