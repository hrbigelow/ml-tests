import torch
from torch.nn import functional as F
import numpy as np


def fix_padding(filter_sz, wanted_padding, is_inverse):
    '''calculate value of 'padding' argument for torch convolutions
    for it to apply the wanted padding.
    '''
    # F.conv_transpose1d adds kernel_size - 1 - p actual padding, for padding=p
    if is_inverse:
        return filter_sz - 1 - wanted_padding  
    else:
        return wanted_padding


def conv(conv_type, input):

    '''use torch's F.conv1d and F.conv_transpose1d to produce the convolution
    (or fractionally strided convolution), with all parameters determined from
    conv_type, on the input'''

    def nest2(x):
        return np.expand_dims(np.expand_dims(x, 0), 0)

    ct = conv_type
    tinput = torch.tensor(nest2(input), dtype=torch.float64)
    tweight = torch.tensor(nest2(ct.filter(do_dilate=False)), dtype=torch.float64)

    # handles strange torch defintion of 'padding' for inverse convolutions
    wanted_pad = max(ct.lpad(), ct.rpad())
    tpad = fix_padding(ct.filter_size(do_dilate=True), wanted_pad, ct.is_inverse)

    if ct.is_inverse:
        conv = F.conv_transpose1d(
                tinput, tweight, bias=None, stride=ct.stride,
                padding=tpad, output_padding=0, groups=1,
                dilation=ct.dilation)
        cmd_string = 'F.conv_transpose1d(input, weight, bias=None, stride={}, ' \
        'padding={}, output_padding=0, groups=1, dilation={})'.format(ct.stride,
                tpad, tdilation)

    else:
        conv = F.conv1d(
                tinput, tweight, bias=None, stride=ct.stride,
                padding=tpad, dilation=ct.dilation,
                groups=1)
        cmd_string = 'F.conv1d(input, weight, bias=None, stride={}, padding={}, ' \
        'dilation={}, groups=1)'.format(ct.stride, tpad, ct.dilation)

    def unnest2(x):
        return np.squeeze(np.squeeze(x, 0), 0)

    nconv = unnest2(conv.numpy())
    return nconv, cmd_string
    
