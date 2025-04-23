import torch
import torch.nn as nn
import torch.nn.functional as F




class CBR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.ReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class BR(nn.Module):
    def __init__(self, nOut):
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.ReLU(nOut)

    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output

class CB(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        output = self.conv(input)
        return output


class C3block(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        if d == 1:
            self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                                  dilation=d)
        else:
            combine_kernel = 2 * d - 1

            self.conv = nn.Sequential(
                nn.Conv2d(nIn, nIn, kernel_size=(combine_kernel, 1), stride=stride, padding=(padding - 1, 0),
                          groups=nIn, bias=False),
                nn.BatchNorm2d(nIn),
                nn.PReLU(nIn),
                nn.Conv2d(nIn, nIn, kernel_size=(1, combine_kernel), stride=stride, padding=(0, padding - 1),
                          groups=nIn, bias=False),
                nn.BatchNorm2d(nIn),
                nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False,
                          dilation=d),
                nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False))

    def forward(self, input):
        output = self.conv(input)
        return output

class DepthwiseSeparableConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        # 深度卷积 (Depthwise Convolution)
        padding = kernel_size // 2 if dilation == 1 else dilation
        self.depthwise_conv = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, dilation=dilation,
                                        groups=in_ch, bias=False)  # groups=in_ch 实现深度卷积
        self.pointwise_conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)  # 逐点卷积 (1x1卷积)

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 深度卷积 -> 逐点卷积 -> BatchNorm -> ReLU
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        return self.relu(x)


class DownDepthwiseSeparableConvBNReLU(DepthwiseSeparableConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return super().forward(x)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))

class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16):
        super(CAB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(
            1)  # AAP 会将输入特征图自适应地压缩到大小为 1x1 的特征图，输出形状为 (batch_size, channels, 1, 1)，每个通道的值会是该通道的所有空间元素的平均值或最大值。
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # AMP
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.relu(self.fc1(avg_pool_out)))
        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.relu(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return x*(self.sigmoid(out))

class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()
        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1=x
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 每个通道执行平均池化和最大池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return x1*(self.sigmoid(x))




# class C3module(nn.Module): #两个输入
#     def __init__(self, In_l,In_r, nOut, D_rate=[2,4,6]):
#         super().__init__()
#         #n = int(nOut / 4)
#         n = int(nOut / 3)
#         #
#         In_l = F.interpolate(In_l, size=In_r.shape[2:], mode='bilinear', align_corners=False)
#         nIn = In_l + In_r
#         self.c1 = C(nIn, n, 1, 1)
#         self.d1 = C3block(n, n, 3, 1, D_rate[0])
#         self.d2 = C3block(n, n, 3, 1, D_rate[1])
#         self.d3 = C3block(n, n, 3, 1, D_rate[2])
#         #self.d4 = C3block(n, n, 3, 1, D_rate[3])
#         self.bn = BR(nOut)
#         self.attention_conv = nn.Conv2d(nOut, 1, kernel_size=1, stride=1, padding=0)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, In_l,In_r):
#         In_l = F.interpolate(In_l, size=In_r.shape[2:], mode='bilinear', align_corners=False)
#         nIn = In_l + In_r
#         output1 = self.c1(nIn)
#         d1 = self.d1(output1)
#         d2 = self.d2(output1)
#         d3 = self.d3(output1)
#         #d4 = self.d4(output1)
#
#         #combine = torch.cat([d1, d2, d3, d4], 1)
#         combine = torch.cat([d1, d2, d3], 1)
#         attention_map = self.attention_conv(combine)
#         attention_map = self.sigmoid(attention_map)
#
#         # combine = input + combine
#         # output = self.bn(combine)
#         #return output
#         return attention_map*In_r

class C3module(nn.Module): #单个输入
    def __init__(self, In, nOut, D_rate=[2,4,6,8]):
        super().__init__()
        n = int(nOut / 4)
        #n = int(nOut / 3)
        #

        self.c1 = C(In, n, 1, 1)
        self.d1 = C3block(n, n, 3, 1, D_rate[0])
        self.d2 = C3block(n, n, 3, 1, D_rate[1])
        self.d3 = C3block(n, n, 3, 1, D_rate[2])
        self.d4 = C3block(n, n, 3, 1, D_rate[3])
        self.bn = BR(nOut)
        #self.attention_conv = nn.Conv2d(nOut, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, In):
        input = In
        output1 = self.c1(In)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)

        combine = torch.cat([d1, d2, d3, d4], 1)
        #combine = torch.cat([d1, d2, d3], 1)
        #attention_map = self.attention_conv(combine)
        # attention_map = self.sigmoid(attention_map)

        combine = input + combine
        output = self.bn(combine)
        return output


class DBRC(nn.Module):
    def __init__(self, In, nOut, D_rate=[2, 4, 6,8], kernel_size=3, dilation=1):
        super().__init__()

        # 深度可分离卷积层
        self.depthwise_separable_conv = DepthwiseSeparableConvBNReLU(In, nOut, kernel_size, dilation)
        self.cab = CAB(nOut)
        self.sab = SAB()
        # C3模块
        #self.c3module = C3module(nOut, nOut, D_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先通过深度可分离卷积
        x = self.depthwise_separable_conv(x)
        x = self.cab(x)
        x = self.sab(x)

        # 然后通过C3模块
        #x = self.c3module(x)

        return x


class Down_DBRC(nn.Module):
    def __init__(self, In, nOut, D_rate=[2, 4, 6,8], kernel_size=3, dilation=1,flag = True):
        super().__init__()

        self.depthwise_separable_conv = DownDepthwiseSeparableConvBNReLU(In, nOut, kernel_size, dilation,flag)
        #self.c3module = C3module(nOut, nOut, D_rate)
        self.cab = CAB(nOut)
        self.sab = SAB()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先通过深度可分离卷积
        x = self.depthwise_separable_conv(x)
        x = self.cab(x)
        x = self.sab(x)
        # 然后通过C3模块
        #x = self.c3module(x)

        return x