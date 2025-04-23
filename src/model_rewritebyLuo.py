from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    ##下采样+CBR

class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


#################如果要修改的话，那就是需要修改RSU模块，
class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        assert height >= 2

        self.conv_in = ConvBNReLU(in_ch, out_ch)
        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        ######先是ConvBNReLU，后每层都是DownConvBNReLU
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]

        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))
            ###########该算法作用：确保de最后一输出通道是out_ch

        #########en最后一层为CBR，d=2，如果我要添加自己的，直接在本层前面再添加一层，并减少循环添加的一层######################
        ##假设我有定义好的vit网络
        ##encode_list.append(vit())
        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))

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


class U2Net(nn.Module):
    def __init__(self, cfg: dict, out_ch: int = 1):
        super().__init__()
        assert "encode" in cfg
        assert "decode" in cfg
        self.encode_num = len(cfg["encode"])

        encode_list = []
        side_list = []
        for c in cfg["encode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
            #根据 c[4] 的值选择创建一个 RSU 对象还是 RSU4F 对象，然后将这个对象添加到 encode_list 列表中。如果 c[4] 为 False，则使用 RSU(*c[:4])。如果 c[4] 为 True，则使用 RSU4F(*c[1:4])。
            # [4, 512, 256, 512, True, False],   # En5  以这里为例，[4]是True，于是En5是一个RSU4F
            #RSU(*c[:4])：如果 c[4] 是 False，则创建一个 RSU 对象，并将 c[:4] 作为参数传递给 RSU 构造函数（* 是用于解包列表作为函数参数的语法）。
            #RSU4F(*c[1:4])：如果 c[4] 不是 False（即为 True），则创建一个 RSU4F 对象，并将 c[1:4] 作为参数传递给 RSU4F 构造函数。

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
            # [5]设置为True，代表着有side输出，即预测输出
        self.encode_modules = nn.ModuleList(encode_list)
        #将encode_list放进容器

        decode_list = []
        for c in cfg["decode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))   ##这个列表存放着对每个解码器最终图像做处理的步骤，得到每层的预测图像out_ch是1
        # side_list中只是添加了对应的卷积核
        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)


    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape

        # collect encode outputs
        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)   ##经过RSU的输出x
            encode_outputs.append(x)
            if i != self.encode_num - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
                ##对RSU进行下采样

        # collect decode outputs
        x = encode_outputs.pop()  ##  这里的x是En6，不会改变
        decode_outputs = [x]  #En6首先被放入decode_outputs
        for m in self.decode_modules:
            x2 = encode_outputs.pop()          ##循环第一次x2是En5 之后依次是En4、En3、En2、En1 ;encode_outputs [1 2 3 4 5]->[1 2 3 4]->[1 2 3]
        #                                                                                                       i=0           1        2
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)    ##将x上采样到x2同样的尺寸，即将En6上采样到En5大小、De5和En4、De4和En3、De3和En2、De2和En1
            x = m(torch.concat([x, x2], dim=1))  #将En5和En6拼接形成De5、De4、De3、De2、De1
            decode_outputs.insert(0, x)  #将x插到decode_outputs最前面，即将De5插入到En6前，之后循环形成De1、De2、De3、De4、De5、En6

        #跳跃连接处操作
         #假设在En3后的跳跃 引入 trans 首先修改循环条件，添加变量i记录循环次数
        # for i, m in enumerate(self.decode_modules):
        #     x2 = encode_outputs.pop()
        #     if i == 2:  # De4是解码的第4层，即对应于En3
        #         en3_output = encode_outputs[-1]  # 获取En3的输出
        #         transformer_output = self.transformer(en3_output)  # 通过transformer处理En3的输出      自己定义transformer层
        #         # 将transformer的输出调整为与x2相同的大小
        #         transformer_output = F.interpolate(transformer_output, size=x2.shape[2:], mode='bilinear',
        #                                            align_corners=False)
        #         x = torch.cat([x, transformer_output], dim=1)  # 将Transformer输出与De4的输入进行拼接
        #     else:
        #         x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        #     x = m(torch.concat([x, x2], dim=1))  # 将En5和En6拼接形成De5、De4、De3、De2、De1
        #     decode_outputs.insert(0, x)
        ##以上为引入trans

        #进入for循环前，encoder_outputs的最后一个即En6已经被弹出，作为decode_outputs的第一个，进入循环后，将En5赋给x2

        #此处可以捋一下过程
        #第一次循环 i = 0: x是En6.x2是En5，将En6上采样到En5的大小拼接形成De5，放入decode_outputs=[De5 En6]
        #第二次循环 i = 1：x是De5，x2是En4，将De5上采样到En4的大小拼接形成De4，放入decode_outputs=[De4 De5 En6]
        #第三次循环 i = 2：x是De4，x2是En3，将De4上采样到En3的大小拼接形成De3，放入decode_outputs=[De3 De4 De5 En6]
        #如果想将En3处通过transformer，则在第三次循环，i=2处做改动
        #
        # collect side outputs

        side_outputs = []
        for m in self.side_modules:  #side_modules中是6个输入通道不同，输出通道为1的卷积核
            x = decode_outputs.pop() #依次是En6、De5、De4、De3、De2、De1
            x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)  #将En6、De5、De4、De3、De2、De1经过对应的卷积核后，恢复成原图片大小，通道为1
            ##m是卷积核，将decode_outputs中的输出经过卷积核，得到输出特征图
            side_outputs.insert(0, x)   #新生成的x放在最前面
            ##将特征图放入side_outputs中 ，形成了side1、side2、side3、side4、side5、side6
        x = self.out_conv(torch.concat(side_outputs, dim=1))  ##对一个通道数6，大小与原图大小相同的图进行输入通道6，输出通道1，卷积核大小1的卷积，形成通道为1的图

        if self.training:
            # do not use torch.sigmoid for amp safe
            return [x] + side_outputs   #计算损失
        else:
            return torch.sigmoid(x)   #卷积结果转为概率值，二分类
            #return [torch.sigmoid(x_i) for x_i in side_outputs]+ [torch.sigmoid(x)]  #如果需要查看各层掩码图像效果  返回6个x + 融合后x

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
