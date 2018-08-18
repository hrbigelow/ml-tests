import torch
from torch.nn import functional as F
import numpy as np
import convtype as ctyp


def get_ct(matrix_sz, filt, stride, padding_type, dilation):
    '''Produce a ConvType object that corresponds with these setting'''
    filt_ctr = len(filt) // 2
    phase = 'LEFTMAX'
    return ctyp.ConvType(matrix_sz, filt, filt_ctr, stride,
            False, phase, dilation, padding_type)


def get_ct_transpose(matrix_sz, filt, stride, padding_type, dilation):
    '''Produce a ConvType object that corresponds with these settings'''
    filt_ctr = len(filt) // 2
    phase = 'LEFTMAX'
    return ctyp.ConvType(matrix_sz, filt, filt_ctr,
            stride, True, phase, dilation, padding_type)


def conv(input, filt, is_inverse, stride, padding_type, dilation):
    '''use torch's F.conv1d and F.conv_transpose1d to produce the convolution
    (or fractionally strided convolution), with all parameters determined from
    conv_type, on the input'''

    def nest2(x):
        return np.expand_dims(np.expand_dims(x, 0), 0)

    tinput = torch.tensor(nest2(input), dtype=torch.float64)
    tweight = torch.tensor(nest2(filt), dtype=torch.float64)

    filt_sz = len(filt) * dilation - dilation
    if padding_type == 'SAME':
        padding = int(filt_sz / 2)
    elif padding_type == 'VALID':
        padding = 0

    if is_inverse:
        conv = F.conv_transpose1d(
                tinput, tweight, bias=None, stride=stride,
                padding=padding, output_padding=0, groups=1,
                dilation=dilation)
        cmd = 'F.conv_transpose1d(input, weight, bias=None, stride={}, ' \
        'padding={}, output_padding=0, groups=1, dilation={})'.format(stride,
                padding, dilation)

    else:
        conv = F.conv1d(
                tinput, tweight, bias=None, stride=stride,
                padding=padding, dilation=dilation,
                groups=1)
        cmd = 'F.conv1d(input, weight, bias=None, stride={}, padding={}, ' \
        'dilation={}, groups=1)'.format(stride, padding, dilation)

    def unnest2(x):
        return np.squeeze(np.squeeze(x, 0), 0)

    nconv = unnest2(conv.numpy())
    return nconv, cmd
    
