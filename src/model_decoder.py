from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.decoders import EMCAD
from typing import Optional
from src.vit_conv import Block_encoder_bottleneck
from src.C3_Block import DBRC,Down_DBRC

#from swin_transformer import SwinTransformer

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ConvModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None):
        super(ConvModule, self).__init__()

        # 卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)

        # 批归一化
        if norm_cfg is not None:
            norm_type = norm_cfg.get('type', 'BN')
            if norm_type == 'BN':
                self.bn = nn.BatchNorm2d(out_channels, momentum=norm_cfg.get('momentum', 0.03),
                                         eps=norm_cfg.get('eps', 0.001))
            else:
                self.bn = nn.Identity()  # No normalization if type is not 'BN'
        else:
            self.bn = nn.Identity()  # No normalization

        # 激活函数
        if act_cfg is not None:
            act_type = act_cfg.get('type', 'ReLU')
            if act_type == 'SiLU':
                self.act = nn.SiLU()
            elif act_type == 'ReLU':
                self.act = nn.ReLU(inplace=True)
            else:
                self.act = nn.Identity()  # No activation if type is not supported
        else:
            self.act = nn.Identity()  # No activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))
    ##下采样+CBR

class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


class GhostConvBNReLU(nn.Module):
    """Ghost卷积替代标准ConvBNReLU"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, ratio=2):
        super().__init__()
        padding = kernel_size // 2 if dilation == 1 else dilation

        # 基础卷积生成部分特征
        init_ch = int(out_ch / ratio)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, kernel_size,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(inplace=True))

        # 廉价操作生成幻影特征
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(init_ch, out_ch - init_ch, kernel_size=3,
                      padding=1, groups=init_ch, bias=False),  # 深度卷积
            nn.BatchNorm2d(out_ch - init_ch),
            nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_conv(x1)
        return torch.cat([x1, x2], dim=1)

class GhostDownConvBNReLU(GhostConvBNReLU):
    """带下采样的Ghost卷积模块"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dilation: int = 1, flag: bool = True, ratio=2):
        super().__init__(in_ch, out_ch, kernel_size, dilation, ratio)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        return super().forward(x)

class GhostUpConvBNReLU(GhostConvBNReLU):
    """带上采样的Ghost卷积模块"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                dilation: int = 1, flag: bool = True, ratio=2):
        # 输入通道需要加倍（拼接特征）
        super().__init__(in_ch*2, out_ch, kernel_size, dilation, ratio)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:],
                              mode='bilinear', align_corners=False)
        concat = torch.cat([x1, x2], dim=1)
        return super().forward(concat)

class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        assert height >= 2

        self.conv_in = ConvBNReLU(in_ch, out_ch)  #ConvBNReLU
        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]  #DownConvBNReLU
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]   #UpConvBNReLU
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))   #DownConvBNReLU
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))   #UpConvBNReLU
        ##########该算法作用：确保de最后一输出通道是out_ch
        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))  #ConvBNReLU

        ##ghost
        # self.conv_in = GhostConvBNReLU(in_ch, out_ch, ratio=2)
        # encode_list = [GhostDownConvBNReLU(out_ch, mid_ch, flag=False)]
        # decode_list = [GhostUpConvBNReLU(mid_ch, mid_ch, flag=False)]
        # for i in range(height - 2):
        #     encode_list.append(GhostDownConvBNReLU(mid_ch, mid_ch))
        #     decode_list.append(GhostUpConvBNReLU(
        #         mid_ch, mid_ch if i < height - 3 else out_ch
        #     ))
        # encode_list.append(GhostConvBNReLU(mid_ch, mid_ch, dilation=2))





        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        return x + x_in


class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

        # self.conv_in = GhostConvBNReLU(in_ch, out_ch)
        # self.encode_modules = nn.ModuleList([GhostConvBNReLU(out_ch, mid_ch),
        #                                      GhostConvBNReLU(mid_ch, mid_ch, dilation=2),
        #                                      GhostConvBNReLU(mid_ch, mid_ch, dilation=4),
        #                                      GhostConvBNReLU(mid_ch, mid_ch, dilation=8)])
        #
        # self.decode_modules = nn.ModuleList([GhostConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
        #                                      GhostConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
        #                                      GhostConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in

class likeLGAG(nn.Module):
    def __init__(self,l_channel,r_channel,out_channel,kernel_size=3,groups=1):
        super(likeLGAG,self).__init__()
        if kernel_size == 1:
            groups = 1
        self.W_l = nn.Sequential(
            nn.Conv2d(l_channel, out_channel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(out_channel)
        )
        self.W_r = nn.Sequential(
            nn.Conv2d(r_channel, out_channel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(out_channel)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(out_channel, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)


    def forward(self, l, r):
        g1 = self.W_l(l)
        x1 = self.W_r(r)
        g1_upsampled = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1_upsampled + x1)  # 单纯的g1+x1
        psi = self.psi(psi)
        return l*psi


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




class CAA(nn.Module):
    """Context Anchor Attention"""

    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
    ):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor



class U2Net(nn.Module): #DBRC编码器，EMCAD解码器,
    def __init__(self, cfg: dict, out_ch: int = 1):
        super().__init__()
        assert "encode" in cfg
        assert "decode" in cfg
        self.encode_num = len(cfg["encode"])
        encode_list = []
        side_list = []
        for c in cfg["encode"]:
            assert len(c) == 6
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))  #encode_list : RSU RSU RSU RSU RSU4F RSU4F
            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))    #side_list:En6  conv2d(512,1,3,1)
        self.encode_modules = nn.ModuleList(encode_list) #encode_modules : RSU RSU RSU RSU RSU4F RSU4F

        decode_list = []
        for c in cfg["decode"]:
            assert len(c) == 6
            decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))  #decode_list: RSU4F RSU RSU RSU RSU

            if c[5] is True:    #side_list: De5 De4 De3 De2 De1
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))   ##这个列表存放着对每个解码器最终图像做处理的步骤，得到每层的预测图像out_ch是1
                #conv2d(512,1,3,1)、conv2d(256,1,3,1)、conv2d(128,1,3,1)、conv2d(64,1,3,1)、conv2d(64,1,3,1)
        # side_list中只是添加了对应的卷积核
        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)
        self.decoder = EMCAD(channels=[512, 512, 256, 128, 64], kernel_sizes=[1, 3, 5], expansion_factor=6,dw_parallel=True, add=True, lgag_ks=3, activation='relu6')
        # self.add1 = likeLGAG(l_channel=64, r_channel=64, out_channel=32, kernel_size=3, groups=32)
        # self.add2 = likeLGAG(l_channel=128,r_channel=128,out_channel=64, kernel_size=3, groups=64)
        # self.add3 = likeLGAG(l_channel=256, r_channel=256,out_channel=128, kernel_size=3, groups=128)
        self.dbrc1 = DBRC(3,32)
        self.dbrc2 = Down_DBRC(32,64)
        self.dbrc3 = Down_DBRC(64, 128)
        self.dbrc4 = Down_DBRC(128, 256)
        self.dbrc5 = Down_DBRC(256, 512)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape

        #将编码器单独表示
        encode_outputs = []
        x1 = self.encode_modules[0](x)  # 经过 RSU1 en1输出  ([2, 64, 288, 288])
        encode_outputs.append(x1)
        dbrc1  = self.dbrc1(x)  #([2, 32, 288, 288])
        dbrc2 = self.dbrc2(dbrc1)   #进行add 图上第一块  ([2, 64, 144, 144])
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2, ceil_mode=True)  # ([2, 64, 144, 144])
        #add1 = self.add1(dbrc2, x1)  # ([2, 64, 144, 144])
        x1 = dbrc2 + x1  # ([2, 64, 144, 144])
        x2 = self.encode_modules[1](x1)  # 经过 RSU2  en2输出  ([2, 128, 144, 144])
        encode_outputs.append(x2)

        x2 = F.max_pool2d(x2, kernel_size=2, stride=2, ceil_mode=True)  # ([2, 128, 72, 72])
        dbrc3 = self.dbrc3(dbrc2)  #([2, 128, 72, 72])
        #add2 = self.add2(dbrc3, x2)  # ([2, 128, 72, 72])
        x2 = dbrc3 + x2  # ([2, 128, 72, 72])
        x3 = self.encode_modules[2](x2)  # 经过 RSU3  en3输出   x3 :([2, 256, 72, 72])
        encode_outputs.append(x3)
        x3 = F.max_pool2d(x3, kernel_size=2, stride=2, ceil_mode=True)  # ([2, 256, 36, 36])
        dbrc4 = self.dbrc4(dbrc3)
        #add3 = self.add3(dbrc4, x3)  # ([2, 256, 36, 36])
        x3 = dbrc4 + x3    #([2, 256, 36, 36])
        x4 = self.encode_modules[3](x3)  # 经过 RSU4  en4输出
        encode_outputs.append(x4)
        dbrc5 = self.dbrc5(dbrc4)  #([2, 512, 18, 18])
        x4 = F.max_pool2d(x4, kernel_size=2, stride=2, ceil_mode=True)  # ([2, 512, 18, 18])
        x4 = x4 + dbrc5
        x5 = self.encode_modules[4](x4)  # 经过 RSU4F  en5输出
        encode_outputs.append(x5)
        x5 = F.max_pool2d(x5, kernel_size=2, stride=2, ceil_mode=True)
        x6 = self.encode_modules[5](x5)  # 经过 RSU4F  en6输出
        encode_outputs.append(x6)

        #在这里组成了decoder
        x = encode_outputs.pop()  ##  这里的x是En6的输出，不会改变

        dec_outs = self.decoder(x, encode_outputs,self.decode_modules) #forward(self, x, skips,decode_modules):  #x是En6 512  skips是encoder 【En1 En2 En3 En4 En5】
        side_outputs = []
        for m in self.side_modules:
                    # #conv2d(512,1,3,1)、conv2d(512,1,3,1)、conv2d(256,1,3,1)、conv2d(128,1,3,1)、conv2d(64,1,3,1)、conv2d(64,1,3,1)
            x = dec_outs.pop()  # 依次是En6、De5、De4、De3、De2、De1的输出
            x = F.interpolate(m(x), size=[h, w], mode='bilinear',
                              align_corners=False)  # 将En6、De5、De4、De3、De2、De1经过对应的卷积核后，恢复成原图片大小，通道为1

            side_outputs.insert(0, x)  # 新生成的x放在最前面  形成了side1、side2、side3、side4、side5、side6
        x = self.out_conv(torch.concat(side_outputs, dim=1))

        if self.training:
            return [x] + side_outputs   #计算损失
        else:
            return torch.sigmoid(x)   #卷积结果转为概率值，二分类


# class U2Net(nn.Module): #正常编码器，EMCAD解码器,
#     def __init__(self, cfg: dict, out_ch: int = 1):
#         super().__init__()
#         assert "encode" in cfg
#         assert "decode" in cfg
#         self.encode_num = len(cfg["encode"])
#         encode_list = []
#         side_list = []
#         for c in cfg["encode"]:
#             assert len(c) == 6
#             encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))  #encode_list : RSU RSU RSU RSU RSU4F RSU4F
#             if c[5] is True:
#                 side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))    #side_list:En6  conv2d(512,1,3,1)
#         self.encode_modules = nn.ModuleList(encode_list) #encode_modules : RSU RSU RSU RSU RSU4F RSU4F
#
#         decode_list = []
#         for c in cfg["decode"]:
#             assert len(c) == 6
#             decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))  #decode_list: RSU4F RSU RSU RSU RSU
#
#             if c[5] is True:    #side_list: De5 De4 De3 De2 De1
#                 side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))   ##这个列表存放着对每个解码器最终图像做处理的步骤，得到每层的预测图像out_ch是1
#                 #conv2d(512,1,3,1)、conv2d(256,1,3,1)、conv2d(128,1,3,1)、conv2d(64,1,3,1)、conv2d(64,1,3,1)
#         # side_list中只是添加了对应的卷积核
#         self.decode_modules = nn.ModuleList(decode_list)
#         self.side_modules = nn.ModuleList(side_list)
#         self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)
#         self.decoder = EMCAD(channels=[512, 512, 256, 128, 64], kernel_sizes=[1, 3, 5], expansion_factor=6,dw_parallel=True, add=True, lgag_ks=3, activation='relu6')
#         #self.encoder = Encoder(channels=[3,32,64,128,256])
#
#     def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
#         _, _, h, w = x.shape
#
#         # collect encode outputs
#         encode_outputs = []
#         for i, m in enumerate(self.encode_modules):
#             x = m(x)
#             encode_outputs.append(x)
#             if i != self.encode_num - 1:
#                 x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
#
#         #在这里组成了decoder
#         x = encode_outputs.pop()  ##  这里的x是En6的输出，不会改变
#
#         dec_outs = self.decoder(x, encode_outputs,self.decode_modules) #forward(self, x, skips,decode_modules):  #x是En6 512  skips是encoder 【En1 En2 En3 En4 En5】
#         side_outputs = []
#         for m in self.side_modules:
#                     # #conv2d(512,1,3,1)、conv2d(512,1,3,1)、conv2d(256,1,3,1)、conv2d(128,1,3,1)、conv2d(64,1,3,1)、conv2d(64,1,3,1)
#             x = dec_outs.pop()  # 依次是En6、De5、De4、De3、De2、De1的输出
#             x = F.interpolate(m(x), size=[h, w], mode='bilinear',
#                               align_corners=False)  # 将En6、De5、De4、De3、De2、De1经过对应的卷积核后，恢复成原图片大小，通道为1
#
#             side_outputs.insert(0, x)  # 新生成的x放在最前面  形成了side1、side2、side3、side4、side5、side6
#         x = self.out_conv(torch.concat(side_outputs, dim=1))
#
#         if self.training:
#             return [x] + side_outputs   #计算损失
#         else:
#             return torch.sigmoid(x)   #卷积结果转为概率值，二分类


# class U2Net(nn.Module): #引入DBRC编码器
#     def __init__(self, cfg: dict, out_ch: int = 1):
#         super().__init__()
#         assert "encode" in cfg
#         assert "decode" in cfg
#         self.encode_num = len(cfg["encode"])
#
#         encode_list = []
#         side_list = []
#         for c in cfg["encode"]:
#             # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
#             assert len(c) == 6
#             encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
#
#             if c[5] is True:
#                 side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
#         self.encode_modules = nn.ModuleList(encode_list)
#
#         decode_list = []
#         for c in cfg["decode"]:
#             # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
#             assert len(c) == 6
#             decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
#
#             if c[5] is True:
#                 side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
#         self.decode_modules = nn.ModuleList(decode_list)
#         self.side_modules = nn.ModuleList(side_list)
#         self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)
#         # self.add1 = likeLGAG(l_channel=64, r_channel=64, out_channel=32, kernel_size=3, groups=32)
#         # self.add2 = likeLGAG(l_channel=128,r_channel=128,out_channel=64, kernel_size=3, groups=64)
#         # self.add3 = likeLGAG(l_channel=256, r_channel=256,out_channel=128, kernel_size=3, groups=128)
#         self.dbrc1 = DBRC(3,32)
#         self.dbrc2 = Down_DBRC(32,64)
#         self.dbrc3 = Down_DBRC(64, 128)
#         self.dbrc4 = Down_DBRC(128, 256)
#         self.dbrc5 = Down_DBRC(256, 512)
#         self.vit4 = Block_encoder_bottleneck("first", 256, 512, 8, 0)
#
#     def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
#         _, _, h, w = x.shape
#         encode_outputs = []
#         x1 = self.encode_modules[0](x)  # 经过 RSU1 en1输出  ([2, 64, 288, 288])
#         encode_outputs.append(x1)
#         dbrc1  = self.dbrc1(x)  #([2, 32, 288, 288])
#         dbrc2 = self.dbrc2(dbrc1)   #进行add 图上第一块  ([2, 64, 144, 144])
#
#         x1 = F.max_pool2d(x1, kernel_size=2, stride=2, ceil_mode=True) #([2, 64, 144, 144])
#
#         x1 = dbrc2 + x1
#         x2 = self.encode_modules[1](x1)  # 经过 RSU2  en2输出  ([2, 128, 144, 144])
#         encode_outputs.append(x2)
#
#         x2 = F.max_pool2d(x2, kernel_size=2, stride=2, ceil_mode=True)  #([2, 128, 72, 72])
#         #l_x2 = l_x2 + add1  #([2, 64, 144, 144])
#         dbrc3 = self.dbrc3(dbrc2)  #([2, 128, 72, 72])
#
#         # add2 = self.add2(dbrc3, x2)  #([2, 128, 72, 72])
#         # x2 = add2 + x2  #([2, 128, 72, 72])
#         x2 = dbrc3 + x2
#         x3 = self.encode_modules[2](x2)  # 经过 RSU3  en3输出   x3 :([2, 256, 72, 72])
#         encode_outputs.append(x3)
#
#         x3 = F.max_pool2d(x3, kernel_size=2, stride=2, ceil_mode=True)   #([2, 256, 36, 36])
#         #l_x3 = l_x3+ add2  #([2, 128, 72, 72])
#         dbrc4 = self.dbrc4(dbrc3)
#
#         # add3 = self.add3(dbrc4, x3)   #([2, 256, 36, 36])
#         # x3 = add3 + x3    #([2, 256, 36, 36])
#         x3 = dbrc4 + x3
#         x4 = self.encode_modules[3](x3)  # 经过 RSU4  en4输出
#         encode_outputs.append(x4)
#         #l_x4 = l_x4 + add3  #([2, 256, 36, 36])
#         #dbrc5 = self.dbrc5(dbrc4)  #([2, 512, 18, 18])
#         dbrc5 = self.vit4(dbrc4)
#
#         x4 = F.max_pool2d(x4, kernel_size=2, stride=2, ceil_mode=True)   #([2, 512, 18, 18])
#
#         x4 = dbrc5 + x4
#         x5 = self.encode_modules[4](x4)  # 经过 RSU4F  en5输出
#         encode_outputs.append(x5)
#         x5 = F.max_pool2d(x5, kernel_size=2, stride=2, ceil_mode=True)
#         x6 = self.encode_modules[5](x5)  # 经过 RSU4F  en6输出
#         encode_outputs.append(x6)
#
#         # collect decode outputs
#         x = encode_outputs.pop()
#         decode_outputs = [x]
#         for m in self.decode_modules:
#             x2 = encode_outputs.pop()
#             x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
#             x = m(torch.concat([x, x2], dim=1))
#             decode_outputs.insert(0, x)
#
#         # collect side outputs
#         side_outputs = []
#         for m in self.side_modules:
#             x = decode_outputs.pop()
#             x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
#             side_outputs.insert(0, x)
#
#         x = self.out_conv(torch.concat(side_outputs, dim=1))
#
#         if self.training:
#             # do not use torch.sigmoid for amp safe
#             return [x] + side_outputs
#         else:
#             return torch.sigmoid(x)

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )



# class U2Net(nn.Module): #UNet
#     def __init__(self,in_channels: int = 3,num_classes: int = 1,bilinear: bool = True,base_c: int = 32):
#         super(U2Net, self).__init__()
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         self.bilinear = bilinear
#         self.in_conv = DoubleConv(3, base_c)
#         self.down1 = Down(base_c, base_c * 2)
#         self.down2 = Down(base_c * 2, base_c * 4)
#         self.down3 = Down(base_c * 4, base_c * 8)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(base_c * 8, base_c * 16 // factor)
#         self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
#         self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
#         self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
#         self.up4 = Up(base_c * 2, base_c, bilinear)
#         self.out_conv = OutConv(base_c, num_classes)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         x1 = self.in_conv(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.out_conv(x)
#
#         if self.training:
#             # do not use torch.sigmoid for amp safe
#             return [x]
#         else:
#             return torch.sigmoid(x)



# class U2Net(nn.Module):  # 引入vit编码器 太大了算不动
#     def __init__(self, cfg: dict, out_ch: int = 1):
#         super().__init__()
#         assert "encode" in cfg
#         assert "decode" in cfg
#         self.encode_num = len(cfg["encode"])
#
#         encode_list = []
#         side_list = []
#         for c in cfg["encode"]:
#             # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
#             assert len(c) == 6
#             encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
#
#             if c[5] is True:
#                 side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
#         self.encode_modules = nn.ModuleList(encode_list)
#
#         decode_list = []
#         for c in cfg["decode"]:
#             # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
#             assert len(c) == 6
#             decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
#
#             if c[5] is True:
#                 side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
#         self.decode_modules = nn.ModuleList(decode_list)
#         self.side_modules = nn.ModuleList(side_list)
#         self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)
#         self.dbrc1 = DBRC(3, 32)
#         self.vit1 = Block_encoder_bottleneck("first", 32, 64, 2, 0)
#         self.vit2 = Block_encoder_bottleneck("first", 64, 128, 4, 0)
#         self.vit3 = Block_encoder_bottleneck("first", 128, 256, 4, 0)
#         self.vit4 = Block_encoder_bottleneck("first", 256, 512, 8, 0)
#
#
#     def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
#         _, _, h, w = x.shape
#         encode_outputs = []
#         x1 = self.encode_modules[0](x)  # 经过 RSU1 en1输出  ([2, 64, 288, 288])
#         encode_outputs.append(x1)
#         dbrc1 = self.dbrc1(x)  # ([2, 32, 288, 288])
#         #dbrc2 = self.dbrc2(dbrc1)  # 进行add 图上第一块  ([2, 64, 144, 144])
#         vit1 = self.vit1(dbrc1)
#         x1 = F.max_pool2d(x1, kernel_size=2, stride=2, ceil_mode=True)  # ([2, 64, 144, 144])
#
#         x1 = vit1 + x1
#         x2 = self.encode_modules[1](x1)  # 经过 RSU2  en2输出  ([2, 128, 144, 144])
#         encode_outputs.append(x2)
#
#         x2 = F.max_pool2d(x2, kernel_size=2, stride=2, ceil_mode=True)  # ([2, 128, 72, 72])
#         # l_x2 = l_x2 + add1  #([2, 64, 144, 144])
#         #dbrc3 = self.dbrc3(dbrc2)  # ([2, 128, 72, 72])
#         vit2 = self.vit2(vit1)
#         # add2 = self.add2(dbrc3, x2)  #([2, 128, 72, 72])
#         # x2 = add2 + x2  #([2, 128, 72, 72])
#         x2 = vit2 + x2
#         x3 = self.encode_modules[2](x2)  # 经过 RSU3  en3输出   x3 :([2, 256, 72, 72])
#         encode_outputs.append(x3)
#
#         x3 = F.max_pool2d(x3, kernel_size=2, stride=2, ceil_mode=True)  # ([2, 256, 36, 36])
#         # l_x3 = l_x3+ add2  #([2, 128, 72, 72])
#         vit3 = self.vit3(vit2)
#
#         # add3 = self.add3(dbrc4, x3)   #([2, 256, 36, 36])
#         # x3 = add3 + x3    #([2, 256, 36, 36])
#         x3 = vit3 + x3
#         x4 = self.encode_modules[3](x3)  # 经过 RSU4  en4输出
#         encode_outputs.append(x4)
#         # l_x4 = l_x4 + add3  #([2, 256, 36, 36])
#         vit4 = self.vit4(vit3)  # ([2, 512, 18, 18])
#
#         x4 = F.max_pool2d(x4, kernel_size=2, stride=2, ceil_mode=True)  # ([2, 512, 18, 18])
#
#         x4 = vit4 + x4
#         x5 = self.encode_modules[4](x4)  # 经过 RSU4F  en5输出
#         encode_outputs.append(x5)
#         x5 = F.max_pool2d(x5, kernel_size=2, stride=2, ceil_mode=True)
#         x6 = self.encode_modules[5](x5)  # 经过 RSU4F  en6输出
#         encode_outputs.append(x6)
#
#         # collect decode outputs
#         x = encode_outputs.pop()
#         decode_outputs = [x]
#         for m in self.decode_modules:
#             x2 = encode_outputs.pop()
#             x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
#             x = m(torch.concat([x, x2], dim=1))
#             decode_outputs.insert(0, x)
#
#         # collect side outputs
#         side_outputs = []
#         for m in self.side_modules:
#             x = decode_outputs.pop()
#             x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
#             side_outputs.insert(0, x)
#
#         x = self.out_conv(torch.concat(side_outputs, dim=1))
#
#         if self.training:
#             # do not use torch.sigmoid for amp safe
#             return [x] + side_outputs
#         else:
#             return torch.sigmoid(x)


####U2net 原版
# class U2Net(nn.Module): #原版
#     def __init__(self, cfg: dict, out_ch: int = 1):
#         super().__init__()
#         assert "encode" in cfg
#         assert "decode" in cfg
#         self.encode_num = len(cfg["encode"])
#
#         encode_list = []
#         side_list = []
#         for c in cfg["encode"]:
#             # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
#             assert len(c) == 6
#             encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
#
#             if c[5] is True:
#                 side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
#         self.encode_modules = nn.ModuleList(encode_list)
#
#         decode_list = []
#         for c in cfg["decode"]:
#             # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
#             assert len(c) == 6
#             decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
#
#             if c[5] is True:
#                 side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
#         self.decode_modules = nn.ModuleList(decode_list)
#         self.side_modules = nn.ModuleList(side_list)
#         self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)
#
#     def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
#         _, _, h, w = x.shape
#
#         # collect encode outputs
#         encode_outputs = []
#         for i, m in enumerate(self.encode_modules):
#             x = m(x)
#             encode_outputs.append(x)
#             if i != self.encode_num - 1:
#                 x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
#
#         # collect decode outputs
#         x = encode_outputs.pop()
#         decode_outputs = [x]
#         for m in self.decode_modules:
#             x2 = encode_outputs.pop()
#             x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
#             x = m(torch.concat([x, x2], dim=1))
#             decode_outputs.insert(0, x)
#
#         # collect side outputs
#         side_outputs = []
#         for m in self.side_modules:
#             x = decode_outputs.pop()
#             x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
#             side_outputs.insert(0, x)
#
#         x = self.out_conv(torch.concat(side_outputs, dim=1))
#
#         if self.training:
#             # do not use torch.sigmoid for amp safe
#             return [x] + side_outputs
#         else:
#             return torch.sigmoid(x)


def u2net_full(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 32, 64, False, False],      # En1
                   [6, 64, 32, 128, False, False],    # En2
                   [5, 128, 64, 256, False, False],   # En3
                   [4, 256, 128, 512, False, False],  # En4
                   [4, 512, 256, 512, True, False],   # En5
                   [4, 512, 256, 512, True, True]],   # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 1024, 256, 512, True, True],   # De5
                   [4, 1024, 128, 256, False, True],  # De4
                   [5, 512, 64, 128, False, True],    # De3
                   [6, 256, 32, 64, False, True],     # De2
                   [7, 128, 16, 64, False, True]]     # De1
    }

    return U2Net(cfg, out_ch)
    #return U2Net(out_ch)

def u2net_lite(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 16, 64, False, False],  # En3
                   [4, 64, 16, 64, False, False],  # En4
                   [4, 64, 16, 64, True, False],  # En5
                   [4, 64, 16, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 128, 16, 64, True, True],  # De5
                   [4, 128, 16, 64, False, True],  # De4
                   [5, 128, 16, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return U2Net(cfg, out_ch)


def convert_onnx(m, save_path):
    m.eval()
    x = torch.rand(1, 3, 288, 288, requires_grad=True)

    # export the model
    torch.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      save_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,
                      opset_version=11)


if __name__ == '__main__':
    # n_m = RSU(height=7, in_ch=3, mid_ch=12, out_ch=3)
    # convert_onnx(n_m, "RSU7.onnx")
    #
    # n_m = RSU4F(in_ch=3, mid_ch=12, out_ch=3)
    # convert_onnx(n_m, "RSU4F.onnx")

    u2net = u2net_full()
    convert_onnx(u2net, "u2net_full.onnx")



# class U2Net(nn.Module): 最开始的版本，双编码器，CDEMA解码器，训练速度慢
#     def __init__(self, cfg: dict, out_ch: int = 1):
#         super().__init__()
#         assert "encode" in cfg
#         assert "decode" in cfg
#         self.encode_num = len(cfg["encode"])
#
#         encode_list = []
#         side_list = []
#         for c in cfg["encode"]:
#             # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
#             assert len(c) == 6
#             encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))  #encode_list : RSU RSU RSU RSU RSU4F RSU4F
#             if c[5] is True:
#                 side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))    #side_list:En6  conv2d(512,1,3,1)
#         self.encode_modules = nn.ModuleList(encode_list) #encode_modules : RSU RSU RSU RSU RSU4F RSU4F
#
#         decode_list = []
#         for c in cfg["decode"]:
#             # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
#             assert len(c) == 6
#             decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))  #decode_list: RSU4F RSU RSU RSU RSU
#
#             if c[5] is True:    #side_list: De5 De4 De3 De2 De1
#                 side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))   ##这个列表存放着对每个解码器最终图像做处理的步骤，得到每层的预测图像out_ch是1
#                 #conv2d(512,1,3,1)、conv2d(256,1,3,1)、conv2d(128,1,3,1)、conv2d(64,1,3,1)、conv2d(64,1,3,1)
#         # side_list中只是添加了对应的卷积核
#         self.decode_modules = nn.ModuleList(decode_list)
#         self.side_modules = nn.ModuleList(side_list)
#         self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)
#         self.decoder = EMCAD(channels=[512, 512, 256, 128, 64], kernel_sizes=[1, 3, 5], expansion_factor=6,dw_parallel=True, add=True, lgag_ks=3, activation='relu6')
#         #self.encoder = Encoder(channels=[3,32,64,128,256])
#
#         # 下面的是需要的
#         self.add1 = likeLGAG(l_channel=64,r_channel=64,out_channel=32, kernel_size=3, groups=32)
#         self.add2 = likeLGAG(l_channel=128,r_channel=128,out_channel=64, kernel_size=3, groups=64)
#         self.add3 = likeLGAG(l_channel=256, r_channel=256,out_channel=128, kernel_size=3, groups=128)
#
#         #下面的是需要的
#         self.CBR1 = DepthwiseSeparableConvBNReLU(3, 32)
#         self.CBR2 = DownDepthwiseSeparableConvBNReLU(32, 64)  # 从 32xHxW 到 64x(H/2)x(W/2)
#         self.CBR3 = DownDepthwiseSeparableConvBNReLU(64, 128)  # 输出 128x(H/4)x(W/4)
#         self.CBR4 = DownDepthwiseSeparableConvBNReLU(128, 256)  # 输出 256x(H/8)x(W/8)
#         self.CBR5 = DownDepthwiseSeparableConvBNReLU(256, 512)
#         self.CAA1 = CAA(64)
#         self.CAA2 = CAA(128)
#         self.CAA3 = CAA(256)
#
#     def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
#         _, _, h, w = x.shape
#
#         # collect encode outputs
#         encode_outputs = []
#
#         #将编码器单独表示
#         x1 = self.encode_modules[0](x)  # 经过 RSU1 en1输出  ([2, 64, 288, 288])
#         l_x1  = self.CBR1(x)  #([2, 32, 288, 288])
#         l_x2 = self.CBR2(l_x1)   #进行add 图上第一块  ([2, 64, 144, 144])
#         encode_outputs.append(x1)
#
#         x1 = F.max_pool2d(x1, kernel_size=2, stride=2, ceil_mode=True) #([2, 64, 144, 144])
#         a = self.CAA1(l_x2)
#         l_x2 = l_x2*a
#         add1 = self.add1(l_x2, x1)  #([2, 64, 144, 144])
#         x1 = add1 + x1  #([2, 64, 144, 144])
#         x2 = self.encode_modules[1](x1)  # 经过 RSU2  en2输出  ([2, 128, 144, 144])
#         encode_outputs.append(x2)
#
#         x2 = F.max_pool2d(x2, kernel_size=2, stride=2, ceil_mode=True)  #([2, 128, 72, 72])
#         #l_x2 = l_x2 + add1  #([2, 64, 144, 144])
#         l_x3 = self.CBR3(l_x2)  #([2, 128, 72, 72])
#         b = self.CAA2(l_x3)
#         l_x3 = l_x3*b
#         add2 = self.add2(l_x3, x2)  #([2, 128, 72, 72])
#         x2 = add2 + x2  #([2, 128, 72, 72])
#         x3 = self.encode_modules[2](x2)  # 经过 RSU3  en3输出   x3 :([2, 256, 72, 72])
#
#         encode_outputs.append(x3)
#
#         x3 = F.max_pool2d(x3, kernel_size=2, stride=2, ceil_mode=True)   #([2, 256, 36, 36])
#         #l_x3 = l_x3+ add2  #([2, 128, 72, 72])
#         l_x4 = self.CBR4(l_x3)
#         c = self.CAA3(l_x4)
#         l_x4 = l_x4*c
#         add3 = self.add3(l_x4, x3)   #([2, 256, 36, 36])
#         x3 = add3 + x3    #([2, 256, 36, 36])
#         x4 = self.encode_modules[3](x3)  # 经过 RSU4  en4输出
#         #l_x4 = l_x4 + add3  #([2, 256, 36, 36])
#         l_x5 = self.CBR5(l_x4)  #([2, 512, 18, 18])
#         encode_outputs.append(x4)
#
#         x4 = F.max_pool2d(x4, kernel_size=2, stride=2, ceil_mode=True)   #([2, 512, 18, 18])
#
#         x4 = x4+l_x5
#         x5 = self.encode_modules[4](x4)  # 经过 RSU4F  en5输出
#         encode_outputs.append(x5)
#         x5 = F.max_pool2d(x5, kernel_size=2, stride=2, ceil_mode=True)
#         x6 = self.encode_modules[5](x5)  # 经过 RSU4F  en6输出
#         encode_outputs.append(x6)
#
#
#         #在这里组成了decoder
#         x = encode_outputs.pop()  ##  这里的x是En6的输出，不会改变
#
#         dec_outs = self.decoder(x, encode_outputs,self.decode_modules) #forward(self, x, skips,decode_modules):  #x是En6 512  skips是encoder 【En1 En2 En3 En4 En5】
#         side_outputs = []
#         for m in self.side_modules:
#                     # #conv2d(512,1,3,1)、conv2d(512,1,3,1)、conv2d(256,1,3,1)、conv2d(128,1,3,1)、conv2d(64,1,3,1)、conv2d(64,1,3,1)
#             x = dec_outs.pop()  # 依次是En6、De5、De4、De3、De2、De1的输出
#             x = F.interpolate(m(x), size=[h, w], mode='bilinear',
#                               align_corners=False)  # 将En6、De5、De4、De3、De2、De1经过对应的卷积核后，恢复成原图片大小，通道为1
#
#             side_outputs.insert(0, x)  # 新生成的x放在最前面  形成了side1、side2、side3、side4、side5、side6
#         x = self.out_conv(torch.concat(side_outputs, dim=1))
#
#         if self.training:
#             return [x] + side_outputs   #计算损失
#         else:
#             return torch.sigmoid(x)   #卷积结果转为概率值，二分类