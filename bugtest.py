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
    lz, rz = zeros[0], zeros[-1]

    negones, = np.where(mask == -1)
    try:
        ln, rn = negones[0], negones[-1]
    except IndexError:
        ln, rn = 100000, -100000

    extra_left = max(lz - ln, 0)
    extra_right = max(rn - rz, 0)

    out_padding = max(extra_left, extra_right)
    trim_left = out_padding - extra_left 
    trim_right = out_padding - extra_right

    print('computed adjustments: ', out_padding, trim_left, trim_right)
    return out_padding, trim_left, trim_right 


def wrap_np(x):
    return torch.tensor(np.expand_dims(np.expand_dims(x, 0), 0),
            dtype=torch.float64)

def conv(input, filt, mask):
    '''use torch's F.conv1d and F.conv_transpose1d to produce the convolution
    (or fractionally strided convolution), with all parameters determined from
    conv_type, on the input'''

    tinput = wrap_np(input)
    tweight = wrap_np(filt)

    out_padding, trim_left, trim_right = compute_out_padding(mask)
    print(type(out_padding))

    conv = F.conv_transpose1d(
            input=tinput, weight=tweight, bias=None, stride=3,
            padding=0, output_padding=out_padding, groups=1,
            dilation=1)
    return conv
    
