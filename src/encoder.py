import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.helpers import named_apply


# def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
#     # activation layer
#     act = act.lower()
#     if act == 'relu':
#         layer = nn.ReLU(inplace)
#     elif act == 'relu6':
#         layer = nn.ReLU6(inplace)
#     else:
#         raise NotImplementedError('activation layer [%s] is not found' % act)
#     return layer
#
# def _init_weights(module, name, scheme=''):
#     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
#         if scheme == 'normal':
#             nn.init.normal_(module.weight, std=.02)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
#         nn.init.constant_(module.weight, 1)
#         nn.init.constant_(module.bias, 0)
#     elif isinstance(module, nn.LayerNorm):
#         nn.init.constant_(module.weight, 1)
#         nn.init.constant_(module.bias, 0)
#
# class LGAG(nn.Module):
#     def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1,
#                  activation='relu'):  # F_l是左边线路，F_g是右边线路  对应到网络中，分别对应谁？  g是EUCB那侧 l是编码器侧
#         super(LGAG, self).__init__()
#
#         if kernel_size == 1:
#             groups = 1
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
#                       bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
#                       bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#         self.activation = act_layer(activation, inplace=True)
#
#         self.init_weights('normal')
#
#     def init_weights(self, scheme=''):
#         named_apply(partial(_init_weights, scheme=scheme), self)
#
#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         g1_upsampled = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
#         psi = self.activation(g1_upsampled + x1)  # 单纯的g1+x1
#         psi = self.psi(psi)
#         return x*psi