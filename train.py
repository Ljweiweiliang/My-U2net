import os
import time
import datetime
from typing import Union, List

import torch
import torch.nn as nn
from torch.utils import data

from src.model_decoder import u2net_full
#from att_unet import u2net_full
#from FCN import u2net_full
#from deeplabv3.deeplabv3_model import u2net_full
#from src.model_yuanban import u2net_full
#from DNANET import u2net_full
#from MINet.MINet import u2net_full
from train_utils import train_one_epoch, evaluate, get_params_groups, create_lr_scheduler
from my_dataset import DUTSDataset
import transforms as T
from thop import profile
#u2net_full.pth
class SODPresetTrain:
    def __init__(self, base_size: int, crop_size: int,
                 hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=True),
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(hflip_prob),
            T.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class SODPresetEval:
    def __init__(self, base_size: int, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=False),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

def print_param_count(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print(f'{name}: {param_count} parameters')
    print(f'Total parameters: {total_params}')

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # train_dataset = DUTSDataset(args.data_path, train=True, transforms=SODPresetTrain([320, 320], crop_size=288))
    # val_dataset = DUTSDataset(args.data_path, train=False, transforms=SODPresetEval([320, 320]))
    train_dataset = DUTSDataset(args.data_path, train=True, transforms=SODPresetTrain([320, 320], crop_size=288))
    val_dataset = DUTSDataset(args.data_path, train=False, transforms=SODPresetEval([320, 320]))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True,
                                        pin_memory=True,
                                        collate_fn=train_dataset.collate_fn)

    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,  # must be 1
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      collate_fn=val_dataset.collate_fn)

    #model = u2net_full()
    model = u2net_full()
    # input1 = torch.randn(2,3, 288, 288)
    # flops, params = profile(model,(input1,))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
    model.to(device)

#512
    # if args.pretrained:
    #     print(f"Loading pretrained weights from {args.pretrained}")
    #     checkpoint = torch.load(args.pretrained, map_location=device)
    #     model.load_state_dict(checkpoint['model'], strict=False)  # 'strict=False' to allow mismatch in layers
    #     print("Pretrained weights loaded successfully.")

    #打印参数量
    print_param_count(model)

    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                       warmup=True, warmup_epochs=2)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    current_mae, current_f1 = 1.0, 0.0
    start_time = time.time()
    no_improvement_count = 0  # 记录没有产生新best权重的epoch次数
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            # 每间隔eval_interval个epoch验证一次，减少验证频率节省训练时间


            # 添加acc等指标
            mae_metric, f1_metric, precision, acc= evaluate(model, val_data_loader, device=device)
            mae_info, f1_info ,pre_info,acc_info= mae_metric.compute(), f1_metric.compute(),precision.compute(),acc.compute()
            print(f"[epoch: {epoch}] val_MAE: {mae_info:.3f} val_maxF1: {f1_info:.3f} val_pre: {pre_info:.3f} val_acc: {acc_info:.3f} ")
            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                write_info = f"[epoch: {epoch}] train_loss: {mean_loss:.4f} lr: {lr:.6f} " \
                             f"MAE: {mae_info:.8f} maxF1: {f1_info:.8f} pre: {pre_info:.8f}  acc: {acc_info:.8f} \n"
                f.write(write_info)

            if current_mae > mae_info and current_f1 < f1_info:
                torch.save(save_file, "save_weights/model_best.pth")
                current_mae, current_f1 = mae_info, f1_info
                no_improvement_count = 0  # reset counter if a new best model is found
            else:
                no_improvement_count += 1  # increment the counter if no improvement

            # save_best
            # if current_mae >= mae_info and current_f1 <= f1_info:
            #     torch.save(save_file, "save_weights/model_best.pth")

        # only save latest 10 epoch weights
        if os.path.exists(f"save_weights/model_{epoch-10}.pth"):
            os.remove(f"save_weights/model_{epoch-10}.pth")

        torch.save(save_file, f"save_weights/model_{epoch}.pth")

        if no_improvement_count >= 30:
            print(f"No improvement in the last 30 epochs. Stopping training at epoch {epoch}.")
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch u2net training")

    parser.add_argument("--data-path", default="./", help="DUTS root")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=2, type=int)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--epochs", default=400, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--eval-interval", default=5, type=int, help="validation interval default 10 Epochs")

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", action='store_true',
                        help="Use torch.cuda.amp for mixed precision training")
    #####新加的预训练权重
    #parser.add_argument("--pretrained", default='save_weights/model_best.pth', help="Path to pretrained model")


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
