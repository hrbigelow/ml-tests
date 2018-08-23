import torch
from torch.nn import functional as F
import numpy as np

def compute_padding(mask):
    '''For input_sz=3, filter_sz=7, padding=VALID,
    ConvType Mask: [-2, -2, -2, -1, 0, -1, 0, -1, 0, -1, -2, -2, -2]
    Torch Mask:    [-2, -2, -2, 0, -1, 0, -1, 0, -2, -2, -2]
    The between-stride positions that are adjacent to the INVALID (-2)
    positions are deleted in the Torch Mask.  So, we want to add these
    in, in the form of output_padding.
    Returns: num_left_twos, num_left_ones, num_right_ones, num_right_twos
    '''
    assert 0 in mask

    lt, lo = 0, 0
    for m in mask:
        if m == -2: lt += 1
        elif m == -1: lo += 1
        else: break

    rt, ro = 0, 0
    for m in reversed(mask):
        if m == -2: rt += 1
        elif m == -1: ro += 1
        else: break

    # print('mask: ', mask)
    # print('computed adjustments: ', lt, lo, ro, rt)

    return lt, lo, ro, rt 


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
    lw, rw = ((filt_sz - 1) // 2), (filt_sz // 2)
    l2, l1, r1, r2 = compute_padding(mask)

    if inv:
        # padding = kernel_size - 1 - padding_arg 
        # padding_arg = kernel_size - 1 - padding
        # where padding is applied to both left and right
        # we want to apply individual padding to the left and right.
        # the strategy will be to apply the larger of the two wanted paddings
        # and then trim the excess

        lneed = lw + l2 + l1
        rneed = rw + r2 + r1
        pad = max(lneed, rneed)
        ltrim = max(rneed - lneed, 0)
        rtrim = max(lneed - rneed, 0)
        pad_arg = filt_sz - 1 - pad

        cmd = 'F.conv_transpose1d({}, {}, None, {}, {}, {}, 1, {})'.format(
                tinput.numpy().astype('i'),
                'weight',
                # tweight.numpy().astype('i'),
                st, pad_arg, 0, dil)
        try:
            conv = F.conv_transpose1d(
                    input=tinput, weight=tweight, bias=None, stride=st,
                    padding=pad_arg, output_padding=0, groups=1,
                    dilation=dil)
        except (TypeError, RuntimeError) as ex:
            conv = torch.tensor([[[]]], dtype=torch.float64)
            cmd = str(ex) + ': ' + cmd

        rind = -rtrim or None
        conv = conv[:,:,ltrim:rind]

    else:
        lneed = lw - l2
        rneed = rw - r2
        pad = max(lneed, rneed)
        ltrim = max(rneed - lneed, 0)
        rtrim = max(lneed - rneed, 0)

        cmd = 'F.conv1d(input, weight, None, {}, {}, {}, 1)'.format(st, pad, dil)
        conv = F.conv1d(tinput, tweight, None, st, pad, dil, 1)

        rind = -rtrim or None
        conv = conv[:,:,ltrim:rind]

    def unnest2(x):
        return np.squeeze(np.squeeze(x, 0), 0)

    nconv = unnest2(conv.numpy())
    return nconv, cmd
    
