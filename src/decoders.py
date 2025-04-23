import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_tf_
#from timm.models.helpers import named_apply
from src.SA import sa_layer
from src.ECA import ECAAttention
from src.CBAM import CBAMLayer
from src.CA import CoordAtt
from src.SE import SE_Block
from timm.models import named_apply


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups  # 每组通道数
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2,
              dw_parallel=True, add=True, activation='relu6'):
    """
    create a series of multi-scale convolution blocks.
    """
    convs = []
    mscb = MSCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                dw_parallel=dw_parallel, add=add, activation=activation)
    convs.append(mscb)
    if n > 1:
        for i in range(1, n):
            mscb = MSCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                        dw_parallel=dw_parallel, add=add, activation=activation)
            convs.append(mscb)
    conv = nn.Sequential(*convs)
    return conv  # n=1 一层MSCB块

class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6',
                 dw_parallel=True):  # 最后一个参数是什么？当为False时，使用残差
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([  # 存储多个深度可分离卷积
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes  # 在MSCB中提到，这里的kernel_sizes时[1 3 5]
        ])

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        # You can return outputs based on what you intend to do with them
        return outputs


class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB)
    """

    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                 add=True, activation='relu6'):
        super(MSCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False  # 如果stride = 1 使用跳跃

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation,
                         dw_parallel=self.dw_parallel)  # msdc保存着MSDC的结果
        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)

        msdc_outs = self.msdc(pout1)
        if self.add == True:  # 将多个尺度输出相加
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:  # 将多个尺度按通道拼接
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))

        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)  # 在框架中没有这个步骤
        x = self.pwc(x)
        return x


class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1,
                 activation='relu'):  # F_l是左边线路，F_g是右边线路  对应到网络中，分别对应谁？  g是EUCB那侧 l是编码器侧
        super(LGAG, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # print("g1 shape:", g1.shape)  # 打印 g1 的形状
        # print("x1 shape:", x1.shape)  # 打印 x1 的形状
        g1_upsampled = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        # print("g1 shape:", g1_upsampled.shape)  # 打印 g1 的形状
        # print("x1 shape:", x1.shape)  # 打印 x1 的形状
        psi = self.activation(g1_upsampled + x1)  # 单纯的g1+x1
        psi = self.psi(psi)
        return x*psi

class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
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
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)
        # 没有乘概率的那一步


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 每个通道执行平均池化和最大池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
        # 没有乘概率的那一步

# class EMCAD(nn.Module):  #论文原版，训练速度慢
#     def __init__(self, channels=[512, 512, 256, 128, 64], kernel_sizes=[1, 3, 5], expansion_factor=6, dw_parallel=True,add=True, lgag_ks=3, activation='relu6'):
#         super(EMCAD, self).__init__()
#         eucb_ks = 3  # kernel size for eucb
#
#         self.mscb6 = MSCBLayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes,expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
#         self.eucb5 = EUCB(in_channels=channels[0], out_channels=channels[0], kernel_size=eucb_ks, stride=eucb_ks // 2)
#         self.lgag5 = LGAG(F_g=channels[0], F_l=channels[0], F_int=channels[0] // 2, kernel_size=lgag_ks,groups=channels[0] // 2)
#
#         self.mscb5 = MSCBLayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes,expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
#         self.eucb4 = EUCB(in_channels=channels[1], out_channels=channels[1], kernel_size=eucb_ks, stride=eucb_ks // 2)
#         self.lgag4 = LGAG(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2, kernel_size=lgag_ks,groups=channels[1] // 2)
#
#         self.mscb4 = MSCBLayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes,expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
#         self.eucb3 = EUCB(in_channels=channels[2], out_channels=channels[2], kernel_size=eucb_ks, stride=eucb_ks // 2)
#         self.lgag3 = LGAG(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2, kernel_size=lgag_ks,groups=channels[2] // 2)
#
#         self.mscb3 = MSCBLayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes,expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
#         self.eucb2 = EUCB(in_channels=channels[3], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks // 2)
#         self.lgag2 = LGAG(F_g=channels[3], F_l=channels[3], F_int=channels[3] // 2, kernel_size=lgag_ks,groups=channels[3] // 2)
#
#         self.mscb2 = MSCBLayer(channels[4], channels[4], n=1, stride=1, kernel_sizes=kernel_sizes,expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
#         self.eucb1 = EUCB(in_channels=channels[4], out_channels=channels[4], kernel_size=eucb_ks, stride=eucb_ks // 2)
#         self.lgag1 = LGAG(F_g=channels[4], F_l=channels[4], F_int=channels[4] // 2, kernel_size=lgag_ks,groups=channels[4] // 2)
#
#         self.mscb1 = MSCBLayer(channels[4], channels[4], n=1, stride=1, kernel_sizes=kernel_sizes,expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
#
#         self.cab6 = CAB(channels[0])  #512
#         self.cab5 = CAB(channels[1])  #512
#         self.cab4 = CAB(channels[2])  #256
#         self.cab3 = CAB(channels[3])  #128
#         self.cab2 = CAB(channels[4])  #64
#         self.cab1 = CAB(channels[4])  #64
#         self.sab = SAB()
#
#
#
#     def forward(self, x, skips,decode_modules):  #x是En6 512  skips是encoder 【En1 En2 En3 En4 En5】
#         #for m in self.decode_modules:   #RSU4F RSU RSU RSU RSU
#         m1 = decode_modules[0]
#         d6 = self.cab6(x) * x   #x是De6
#         d6 = self.sab(d6) * d6
#         d6 = self.mscb6(d6)
#         x_de6 = d6
#         d6 = self.eucb5(d6)
#
#         x5 = self.lgag5(g=d6, x=skips[4])
#
#         x_5 = F.interpolate(d6, size=x5.shape[2:], mode='bilinear', align_corners=False)
#
#         d5 = m1(torch.concat([x_5, x5], dim=1))  #De5出来的
#
#
#         m2 = decode_modules[1]
#         d5 = self.cab5(d5) * d5
#         d5 = self.sab(d5) * d5
#         d5 = self.mscb5(d5)
#         x_de5 = d5
#         d5 = self.eucb4(d5)
#         x4 = self.lgag4(g=d5, x=skips[3])
#         x_4 = F.interpolate(d5, size=x4.shape[2:], mode='bilinear', align_corners=False)
#         d4 = m2(torch.concat([x_4, x4], dim=1))  # De4出来的
#
#         m3 = decode_modules[2]
#         d4 = self.cab4(d4) * d4
#         d4 = self.sab(d4) * d4
#         d4 = self.mscb4(d4)
#         x_de4 = d4
#         d4 = self.eucb3(d4)
#         x3 = self.lgag3(g=d4, x=skips[2])
#         x_3 = F.interpolate(d4, size=x3.shape[2:], mode='bilinear', align_corners=False)
#         d3 = m3(torch.concat([x_3, x3], dim=1))  # De3出来的
#
#         m4 = decode_modules[3]
#         d3 = self.cab3(d3) * d3
#         d3 = self.sab(d3) * d3
#         d3 = self.mscb3(d3)
#         x_de3 = d3
#         d3 = self.eucb2(d3)
#         x2 = self.lgag2(g=d3, x=skips[1])
#         x_2 = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=False)
#         d2 = m4(torch.concat([x_2, x2], dim=1))  # De2出来的
#
#         m5 = decode_modules[4]
#         d2 = self.cab2(d2) * d2
#         d2 = self.sab(d2) * d2
#         d2 = self.mscb2(d2)
#         x_de2 = d2
#         d2 = self.eucb1(d2)
#         x1 = self.lgag1(g=d2, x=skips[0])
#         x_1 = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=False)
#         d1 = m5(torch.concat([x_1, x1], dim=1))  # De1出来的
#         d1 = self.cab1(d1) * d1
#         d1 = self.sab(d1) * d1
#         x_de1 = self.mscb1(d1)
#
#         #return [x_de6,x_de5,x_de4,x_de3,x_de2,x_de1]
#         return [x_de1, x_de2, x_de3, x_de4, x_de5, x_de6]


class EMCAD(nn.Module):
    def __init__(self, channels=[512, 512, 256, 128, 64], kernel_sizes=[1, 3, 5], expansion_factor=6, dw_parallel=True,
                 add=True, lgag_ks=3, activation='relu6'):
        super(EMCAD, self).__init__()
        eucb_ks = 3  # kernel size for eucb
        self.eucb5 = EUCB(in_channels=channels[0], out_channels=channels[0], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgag5 = LGAG(F_g=channels[0], F_l=channels[0], F_int=channels[0] // 2, kernel_size=lgag_ks,
                          groups=channels[0] // 2)
        self.eucb4 = EUCB(in_channels=channels[1], out_channels=channels[1], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgag4 = LGAG(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2, kernel_size=lgag_ks,
                          groups=channels[1] // 2)
        self.eucb3 = EUCB(in_channels=channels[2], out_channels=channels[2], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgag3 = LGAG(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2, kernel_size=lgag_ks,
                          groups=channels[2] // 2)
        self.eucb2 = EUCB(in_channels=channels[3], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgag2 = LGAG(F_g=channels[3], F_l=channels[3], F_int=channels[3] // 2, kernel_size=lgag_ks,
                          groups=channels[3] // 2)
        self.eucb1 = EUCB(in_channels=channels[4], out_channels=channels[4], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgag1 = LGAG(F_g=channels[4], F_l=channels[4], F_int=channels[4] // 2, kernel_size=lgag_ks,
                          groups=channels[4] // 2)
        # self.mscb6 = MSCBLayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes,
        #                        expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
        #                        activation=activation)
        # self.mscb5 = MSCBLayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes,expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        # self.mscb4 = MSCBLayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes,
        #                        expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
        #                        activation=activation)
        # self.mscb3 = MSCBLayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes,
        #                        expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
        #                        activation=activation)
        # self.mscb2 = MSCBLayer(channels[4], channels[4], n=1, stride=1, kernel_sizes=kernel_sizes,
        #                        expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
        #                        activation=activation)
        # self.mscb1 = MSCBLayer(channels[4], channels[4], n=1, stride=1, kernel_sizes=kernel_sizes,
        #                        expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
        #                        activation=activation)
        # self.cab6 = CAB(channels[0])  # 512
        # self.cab5 = CAB(channels[1])  #512
        # self.cab4 = CAB(channels[2])  #256
        # self.cab3 = CAB(channels[3])  #128
        # self.cab2 = CAB(channels[4])  #64
        # self.cab1 = CAB(channels[4])  #64
        # self.sab = SAB()
        # self.ca6 = sa_layer(channels[0],32)
        # self.ca5 = sa_layer(channels[1],32)
        # self.ca4 = sa_layer(channels[2],32)
        # self.ca3 = sa_layer(channels[3],32)
        # self.ca2 = sa_layer(channels[4],32)
        # self.ca1 = sa_layer(channels[4],32)
        #单独eca模块的使用，需要打开上面的ca系列，训练时忘记注释了
        self.eca6 = ECAAttention(channels[0])
        self.eca5 = ECAAttention(channels[1])
        self.eca4 = ECAAttention(channels[2])
        self.eca3 = ECAAttention(channels[3])
        self.eca2 = ECAAttention(channels[4])
        self.eca1 = ECAAttention(channels[4])
        # self.cbam6 = CBAMLayer(channels[0])
        # self.cbam5 = CBAMLayer(channels[1])
        # self.cbam4 = CBAMLayer(channels[2])
        # self.cbam3 = CBAMLayer(channels[3])
        # self.cbam2 = CBAMLayer(channels[4])
        # self.cbam1 = CBAMLayer(channels[4])
        # self.ca6 = CoordAtt(channels[0],channels[0])
        # self.ca5 = CoordAtt(channels[1], channels[1])
        # self.ca4 = CoordAtt(channels[2], channels[2])
        # self.ca3 = CoordAtt(channels[3], channels[3])
        # self.ca2 = CoordAtt(channels[4], channels[4])
        # self.ca1 = CoordAtt(channels[4], channels[4])
        # self.se6 = SE_Block(channels[0])
        # self.se5 = SE_Block(channels[1])
        # self.se4 = SE_Block(channels[2])
        # self.se3 = SE_Block(channels[3])
        # self.se2 = SE_Block(channels[4])
        # self.se1 = SE_Block(channels[4])
    def forward(self, x, skips, decode_modules):  # x是En6 512  skips是encoder 【En1 En2 En3 En4 En5】
        # for m in self.decode_modules:   #RSU4F RSU RSU RSU RSU
        m1 = decode_modules[0]
        #d6 = self.ca6(x,32)
        d6 = self.eca6(x)
        #d6 = self.se6(x)
        #d6 = self.cbam6(x)
        #d6 = self.ca6(x)
        # d6 = self.cab6(x) * x  # x是De6
        # d6 = self.sab(d6) * d6
        # d6 = self.mscb6(d6)
        x_de6 = d6
        #x_de6 = x
        d6 = self.eucb5(d6)
        #d6 = self.eucb5(d6)
        x5 = self.lgag5(g=d6, x=skips[4])
        x_5 = F.interpolate(d6, size=x5.shape[2:], mode='bilinear', align_corners=False)
        d5 = m1(torch.concat([x_5, x5], dim=1))  # De5出来的
        m2 = decode_modules[1]
        #d5 = self.ca5(d5,32)
        d5 = self.eca5(d5)
        #d5 = self.se5(d5)
        #d5 = self.cbam5(d5)
        #d5 = self.ca5(d5)
        # d5= self.cab5(d5) * d5
        # d5 = self.sab(d5) * d5
        # d5 = self.mscb5(d5)
        x_de5 = d5
        d5 = self.eucb4(d5)
        x4 = self.lgag4(g=d5, x=skips[3])
        x_4 = F.interpolate(d5, size=x4.shape[2:], mode='bilinear', align_corners=False)
        d4 = m2(torch.concat([x_4, x4], dim=1))  # De4出来的

        m3 = decode_modules[2]
        #d4 = self.ca4(d4,32)
        d4 = self.eca4(d4)  #256
        #d4 = self.se4(d4)
        #d4 = self.cbam4(d4)
        #d4 = self.ca4(d4)
        # d4 = self.cab4(d4) * d4
        # d4 = self.sab(d4) * d4
        # d4 = self.mscb4(d4)
        x_de4 = d4
        d4 = self.eucb3(d4)
        x3 = self.lgag3(g=d4, x=skips[2])
        x_3 = F.interpolate(d4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d3 = m3(torch.concat([x_3, x3], dim=1))  # De3出来的

        m4 = decode_modules[3]
        #d3 = self.ca3(d3,32)
        d3 = self.eca3(d3)
        #d3 = self.se3(d3)
        #d3 = self.cbam3(d3)
        #d3 = self.ca3(d3)
        # d3 = self.cab3(d3) * d3
        # d3 = self.sab(d3) * d3
        # d3 = self.mscb3(d3)
        x_de3 = d3
        d3 = self.eucb2(d3)
        x2 = self.lgag2(g=d3, x=skips[1])
        x_2 = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d2 = m4(torch.concat([x_2, x2], dim=1))  # De2出来的

        m5 = decode_modules[4]
        #d2 = self.ca2(d2,32)
        d2 = self.eca2(d2)
        #d2 = self.se2(d2)
        #d2 = self.cbam2(d2)
        #d2 = self.ca2(d2)
        # d2 = self.cab2(d2) * d2
        # d2 = self.sab(d2) * d2
        # d2 = self.mscb2(d2)
        x_de2 = d2
        d2 = self.eucb1(d2)
        x1 = self.lgag1(g=d2, x=skips[0])
        x_1 = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d1 = m5(torch.concat([x_1, x1], dim=1))  # De1出来的
        #d1 = self.ca1(d1,32)
        d1 = self.eca1(d1)
        #d1 = self.se1(d1)
        #d1 = self.cbam1(d1)
        #d1 = self.ca1(d1)
        # d1 = self.cab1(d1) * d1
        # d1 = self.sab(d1) * d1
        # x_de1 = self.mscb1(d1)
        x_de1 = d1

        # return [x_de6,x_de5,x_de4,x_de3,x_de2,x_de1]
        return [x_de1, x_de2, x_de3, x_de4, x_de5, x_de6]








