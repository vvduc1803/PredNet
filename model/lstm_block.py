import torch
import math
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(ConvLSTMCell, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        kernel_size = _pair(kernel_size)  # (3, 3)
        stride = _pair(stride)  # (1, 1)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels  # 6
        self.out_channels = out_channels  # 3
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_h = tuple(
            k // 2 for k, s, p, d in zip(kernel_size, stride, padding, dilation))
        # print(self.padding_h)   # (1, 1)
        self.dilation = dilation
        self.groups = groups
        self.weight_i = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size))  # [12, 6, 3, 3]
        # input(self.weight_ih.shape)
        self.weight_g = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size))
        self.weight_f = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size))  # [12, 3, 3, 3]
        self.weight_o = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size))  # [9, 3, 3, 3]
        if bias:
            # print(1)
            self.bias_i = Parameter(torch.Tensor(out_channels))  # 12
            self.bias_g = Parameter(torch.Tensor(out_channels))  # 12
            # input(out_channels)
            self.bias_f = Parameter(torch.Tensor(out_channels))  # 12
            self.bias_o = Parameter(torch.Tensor(out_channels))  # 9
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            self.register_parameter('bias_ch', None)

        self.register_buffer('wc_blank', torch.zeros(1, 1, 1, 1))  # dont change
        self.reset_parameters()

    def reset_parameters(self):
        n = 4 * self.in_channels  # 12
        for k in self.kernel_size:
            n *= k

        stdv = 1. / math.sqrt(n)
        self.weight_i.data.uniform_(-stdv, stdv)
        self.weight_g.data.uniform_(-stdv, stdv)
        self.weight_f.data.uniform_(-stdv, stdv)
        self.weight_o.data.uniform_(-stdv, stdv)
        if self.bias_i is not None:
            self.bias_i.data.uniform_(-stdv, stdv)
            self.bias_g.data.uniform_(-stdv, stdv)
            self.bias_f.data.uniform_(-stdv, stdv)
            self.bias_o.data.uniform_(-stdv, stdv)

    def forward(self, inputs, c_current):

        i = F.conv2d(inputs, self.weight_i, self.bias_i, self.stride, self.padding_h, self.dilation, self.groups)
        f = F.conv2d(inputs, self.weight_f, self.bias_f, self.stride, self.padding_h, self.dilation, self.groups)
        g = F.conv2d(inputs, self.weight_g, self.bias_g, self.stride, self.padding_h, self.dilation, self.groups)
        o = F.conv2d(inputs, self.weight_o, self.bias_o, self.stride, self.padding_h, self.dilation, self.groups)

        i = F.sigmoid(i)
        f = F.sigmoid(f)
        g = F.tanh(g)
        o = F.sigmoid(o)

        c_next = f * c_current + i * g
        r_next = o * F.tanh(c_next)

        return r_next, c_next
