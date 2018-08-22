import torch
from torch.nn import functional as F
import numpy as np
import convtype as ctyp

def compute_out_padding(mask):
    '''For input_sz=3, filter_sz=7, padding=VALID,
    ConvType Mask: [-2, -2, -2, -1, 0, -1, 0, -1, 0, -1, -2, -2, -2]
    Torch Mask:    [-2, -2, -2, 0, -1, 0, -1, 0, -2, -2, -2]
    The between-stride positions that are adjacent to the INVALID (-2)
    positions are deleted in the Torch Mask.  So, we want to add these
    in, in the form of output_padding.
    '''
    assert 0 in mask

    zeros, = np.where(mask == 0)
    lz, rz = int(zeros[0]), int(zeros[-1])

    negones, = np.where(mask == -1)
    try:
        ln, rn = int(negones[0]), int(negones[-1])
    except IndexError:
        ln, rn = 100000, -100000

    extra_left = max(lz - ln, 0)
    extra_right = max(rn - rz, 0)

    out_padding = max(extra_left, extra_right)
    trim_left = out_padding - extra_left 
    trim_right = out_padding - extra_right

    # print('computed adjustments: ', out_padding, trim_left, trim_right)
    return out_padding, trim_left, trim_right 

def wrap_np(x):
    return torch.tensor(np.expand_dims(np.expand_dims(x, 0), 0),
            dtype=torch.float64)

def conv(input, mask, filt, inv, st, phase, padding_type, dil):
    '''use torch's F.conv1d and F.conv_transpose1d to produce the convolution
    (or fractionally strided convolution), with all parameters determined from
    conv_type, on the input'''

    tinput = wrap_np(input)
    tweight = wrap_np(filt)

    filt_sz = len(filt) + (len(filt) - 1) * (dil - 1)

    if padding_type == 'SAME':
        padding = (filt_sz) // 2
        if filt_sz % 2 == 0:
            trim_left = 1 # tweak to produce unequal left/right padding
        else:
            trim_left = 0
    elif padding_type == 'VALID':
        padding = 0
        trim_left = 0

    if inv:
        # padding = kernel_size - 1 - padding_arg 
        # padding_arg = kernel_size - 1 - padding
        padding_arg = filt_sz - 1 - padding
        # padding_arg = padding
        # padding_arg = 0
        out_padding, trim_left, trim_right = compute_out_padding(mask)

        cmd = 'F.conv_transpose1d(' \
                'input={}, weight={}, bias=None, stride={}, ' \
                'padding={}, output_padding={}, groups=1, ' \
                'dilation={})'.format(
                        tinput.numpy().astype('i'),
                        'weight',
                        # tweight.numpy().astype('i'),
                        st, padding_arg, out_padding, dil)
        try:
            conv = F.conv_transpose1d(
                    input=tinput, weight=tweight, bias=None, stride=st,
                    padding=padding_arg, output_padding=out_padding, groups=1,
                    dilation=dil)
        except TypeError as te:
            conv = torch.tensor([[[]]], dtype=torch.float64)
            cmd = 'Not executed. Attempted: ' + cmd + ' with exception: ' + str(te)

        # conv = conv[:,:,trim_left:-trim_right]

    else:
        cmd = 'F.conv1d(input, weight, bias=None, stride={}, padding={}, ' \
        'dilation={}, groups=1)'.format(st, padding, dil)
        conv = F.conv1d(
                tinput, tweight, bias=None, stride=st,
                padding=padding, dilation=dil,
                groups=1)
        conv = conv[:,:,trim_left:]

    def unnest2(x):
        return np.squeeze(np.squeeze(x, 0), 0)

    nconv = unnest2(conv.numpy())
    return nconv, cmd
    
