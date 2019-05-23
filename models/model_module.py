import math
import random
from itertools import repeat
import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn


class PruningModule(Module):
    def prune_by_percentile(self, resample=False, reinit=False,**kwargs):
        """
        Note:
             The pruning percentile is based on all layer's parameters concatenated
        Args:
            q (float): percentile in float
            **kwargs: may contain `cuda`
        """
        # Calculate percentile value
        for name, p in self.named_parameters():
            # We do not prune bias term
            if 'bias' in name or 'mask' in name:
                continue
            tensor = p.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values

            if name.startswith("fc"):
                new_q = 20
            elif name.startswith("conv"):
                new_q = 10
            else:
                raise ValueError(name)
            percentile_value = np.percentile(abs(alive), new_q)
            for _name, module in self.named_modules():
                if name.split(".")[0]  == _name:
                    module.prune(threshold=percentile_value, resample=resample, reinit=reinit)



    def prune_by_std(self, s=0.25):
        """
        Note that `s` is a quality parameter / sensitivity value according to the paper.
        According to Song Han's previous paper (Learning both Weights and Connections for Efficient Neural Networks),
        'The pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer’s weights'
        I tried multiple values and empirically, 0.25 matches the paper's compression rate and number of parameters.
        Note : In the paper, the authors used different sensitivity values for different layers.
        """
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold : {threshold} for layer {name}')
                module.prune(threshold)


class MaskedLinear(Module):
    r"""Applies a masked linear transformation to the incoming data: :math:`y = (A * M)x + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.
    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
        mask: the unlearnable mask for the weight.
            It has the same shape as weight (out_features x in_features)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        # Initialize the mask with 1
        self.mask = Parameter(torch.ones([out_features, in_features]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

    def prune(self, threshold, resample, reinit=False):
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        if resample:
            new_mask = np.random.permutation(new_mask)
        # Apply new weight and mask
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

        if reinit:
            nn.init.xavier_uniform_(self.weight)
            self.weight.data = self.weight.data * self.mask.data


class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(MaskedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.groups = groups

        self.weight_prune = Parameter(torch.Tensor(
            self.out_channels, self.in_channels // self.groups, *self.kernel_size))
        self.mask  =Parameter(torch.ones([self.out_channels, self.in_channels // self.groups, *self.kernel_size]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight_prune, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_prune)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.conv2d(input, self.weight_prune, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def prune(self, threshold, reinit=False):
        weight_dev = self.weight_prune.device
        mask_dev = self.mask.device
        # Convert Tensors to numpy and calculate
        tensor = self.weight_prune.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        # Apply new weight and mask
        self.weight_prune.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

        if reinit:
            nn.init.xavier_uniform_(self.weight_prune)
            self.weight_prune.data = self.weight_prune.data * self.mask.data


def _ntuple(n):
    def parse(x):
        if isinstance(x, tuple):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)