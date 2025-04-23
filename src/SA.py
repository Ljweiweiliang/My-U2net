import torch
import torch.nn as nn
import torch.nn.functional as F

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=32):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Use nn.Parameter-like behavior with regular tensor initialization
        self.cweight = torch.zeros(1, channel // (2 * groups), 1, 1, requires_grad=True)
        self.cbias = torch.ones(1, channel // (2 * groups), 1, 1, requires_grad=True)
        self.sweight = torch.zeros(1, channel // (2 * groups), 1, 1, requires_grad=True)
        self.sbias = torch.ones(1, channel // (2 * groups), 1, 1, requires_grad=True)

        # Initialize weights manually
        nn.init.zeros_(self.cweight)
        nn.init.ones_(self.cbias)
        nn.init.zeros_(self.sweight)
        nn.init.ones_(self.sbias)

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    # def forward(self, x,groups):
    #     device = x.device
    #     b, c, h, w = x.shape
    #
    #     x = x.reshape(b * groups, -1, h, w)
    #     x_0, x_1 = x.chunk(2, dim=1)
    #
    #     # channel attention
    #     xn = self.avg_pool(x_0)
    #     xn = xn.to(device)
    #     xn = self.cweight * xn + self.cbias
    #     xn = x_0 * self.sigmoid(xn)
    #
    #     # spatial attention
    #     xs = self.gn(x_1)
    #     xs = self.sweight * xs + self.sbias
    #     xs = x_1 * self.sigmoid(xs)
    #
    #     # concatenate along channel axis
    #     out = torch.cat([xn, xs], dim=1)
    #     out = out.reshape(b, -1, h, w)
    #
    #     out = self.channel_shuffle(out, 2)
    #     return out

    def forward(self, x, groups):
        device = x.device  # 获取输入 x 的设备

        b, c, h, w = x.shape
        x = x.reshape(b * groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = xn.to(device)  # 确保 xn 在相同的设备上
        xn = self.cweight.to(device) * xn + self.cbias.to(device)  # 确保 self.cweight 和 self.cbias 在相同的设备上
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = xs.to(device)  # 确保 xs 在相同的设备上
        xs = self.sweight.to(device) * xs + self.sbias.to(device)  # 确保 self.sweight 和 self.sbias 在相同的设备上
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out
