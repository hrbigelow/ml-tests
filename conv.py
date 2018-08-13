# Experiments with convolutions using a proof-of-principle approach. 

# In these tests, a convolution is implemented as a single, square matrix
# multiplication of the input.  the main concepts of stride, filter width,
# dilation, fractional stride, left and right padding are all implemented
# simply by designing the matrix, in particular, choosing which elements are
# zeros.  This approach is inspired by Naoki Shibuya's Medium article,
# https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0

# Because the matrix is square, the output is the same size as the input, even
# for cases of non-integer stride or fractional stride (up-sampling).  The
# one-to-one correspondence between input and output elements makes explicit
# the distinction between striding and padding effects.  It is defined as
# follows:

# A single filter element is chosen as the 'reference' element using
# filt_ref_index.  The output of a convolution is then assigned to the input
# element that is covered by this reference element.

# To recover the final convolution, a mask is provided for the output elements,
# with values VALID_VAL, SKIP_VAL, and INVALID_VAL.  SKIP_VAL indicates that
# the position was skipped due to stride.  INVALID_VAL indicates the position
# wasn't a valid convolution due to the filter being off the end of the padded
# input.

# The final convolution is then: conv = conv_raw[mask # == VALID_VAL] 

import numpy as np
import torch
from torch.nn import functional as F
import itertools
from sys import stderr

VALID_VAL = 0 
SKIP_VAL = -1
INVALID_VAL = -2

def remove_vals(x, *vals):
    for v in vals:
        x = x[x != v]
    return x


def dilate_index(i, dilation):
    return i * (dilation + 1)


def dilate_array(x, dilation, fill_value):
    sz = len(x)
    assert dilation >= 0

    d = np.full([sz + (sz - 1) * dilation], fill_value=fill_value)
    for p in range(sz):
        dp = dilate_index(p, dilation)
        d[dp] = x[p]
        
    return d


class ConvType(object):

    def __init__(self, filt, filt_ref_index, stride, is_inverse, phase, dilation, lpad, rpad):
        assert stride >= 1
        assert phase < stride
        assert dilation >= 0
        assert filt_ref_index >= 0 and filt_ref_index < len(filt)
        assert lpad >= 0 
        assert rpad >= 0

        self.filt = np.array(filt, dtype=np.float)
        self.filt_ref_index = filt_ref_index 
        self.stride = stride
        self.is_inverse = is_inverse
        self.dilation = dilation
        self.phase = phase
        self._lpad = lpad
        self._rpad = rpad
        
    def lpad(self):
        '''truncates any left padding that is unnecessary to produce a valid
        convolution at the first input position 
        '''
        return min(self._lpad, self.filter_ref_index(do_dilate=True))

    def rpad(self):
        return min(self._rpad,
                dilate_index(len(self.filt) - 1, self.dilation) - \
                        self.filter_ref_index(do_dilate=True))


    def filter(self, do_dilate):
        if do_dilate:
            return dilate_array(self.filt, self.dilation, 0)
        else:
            return self.filt

    def filter_size(self, do_dilate):
        return len(self.filter(do_dilate))


    def filter_ref_index(self, do_dilate):
        if do_dilate:
            return dilate_index(self.filt_ref_index, self.dilation)
        else:
            return self.filt_ref_index


    def valid_pos(self, input_sz, pos):
        '''
        if the filter's reference element is placed at pos, 
        return true if the filter completely overlaps the padded input
        '''
        fi = self.filter_ref_index(do_dilate=True)
        fs = self.filter_size(do_dilate=True)
        filt_beg = pos - fi 
        filt_end = pos + fs - fi
        input_beg = -self.lpad()
        input_end = input_sz + self.rpad()
        return input_beg <= filt_beg and filt_end <= input_end 


    def conv_mat(self, input_sz):
        '''
        outputs:
        mat (input_sz x input_sz)
        mask (input_sz)

        use as:
        conv_raw = np.matmul(mat, input) 
        conv = conv_raw[mask == VALID_VAL] 
        '''
        # check arguments
        assert input_sz >= 0

        do_dilate = True
        fc = self.filter_ref_index(do_dilate)
        fci = self.filter_size(do_dilate) - fc 
        filt = self.filter(do_dilate)

        mat = np.zeros([input_sz, input_sz])
        mask = np.zeros([input_sz])

        for r in range(input_sz):
            if (not self.is_inverse) and r % self.stride != self.phase:
                mat[r,:] = np.full([input_sz], 0)
                mask[r] = SKIP_VAL
                continue

            c = r
            if self.valid_pos(input_sz, c):
                min_off = min(fc, c)
                ub_off = min(fci, input_sz - c) 
                for o in range(-min_off, ub_off):
                    mat[r,c + o] = filt[fc + o]
                mask[r] = VALID_VAL
            else:
                mat[r,:] = np.full([input_sz], 0)
                mask[r] = INVALID_VAL

        return mat, mask

    def conv(self, input):
        # apply stride to the input rather than the output when doing the
        # inverse.
        if self.is_inverse:
            processed = dilate_array(input, self.stride - 1, 0)
        else:
            processed = input 

        mat, mask = self.conv_mat(len(processed))
        conv = np.matmul(mat, processed)

        return input, processed, conv, mask 



def torch_padding(filter_sz, wanted_padding, is_inverse):
    '''calculate value of 'padding' argument for torch convolutions.
    '''
    wing_sz = f_sz // 2

    # What to do with this?
    # assert ct.lpad == ct.rpad

    # F.conv_transpose1d adds kernel_size - 1 - p actual padding, for padding=p
    if is_inverse:
        tpad = filter_sz - 1 - wanted_padding  
    else:
        tpad = wanted_padding

    return tpad


def torch_conv(conv_type, input):
    def nest2(x):
        return np.expand_dims(np.expand_dims(x, 0), 0)

    ct = conv_type
    tinput = torch.tensor(nest2(input), dtype=torch.float64)
    tweight = torch.tensor(nest2(ct.filter(do_dilate=False)), dtype=torch.float64)
    input_sz = len(input)

    # handles strange torch defintion of 'padding' for inverse convolutions
    tpad = torch_padding(ct.filter_size(do_dilate=True), ct.lpad(), ct.is_inverse)

    # by convention, torch minimum dilation is 1
    tdilation = ct.dilation + 1

    # torch minimum dilation = 1 by convention
    if ct.is_inverse:
        conv = F.conv_transpose1d(
                tinput, tweight, bias=None, stride=ct.stride,
                padding=tpad, output_padding=0, groups=1,
                dilation=ct.dilation + 1)

    else:
        conv = F.conv1d(
                tinput, tweight, bias=None, stride=ct.stride,
                padding=tpad, dilation=ct.dilation + 1,
                groups=1)

    def unnest2(x):
        return np.squeeze(np.squeeze(x, 0), 0)

    nconv = unnest2(conv.numpy())
    return nconv
    

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Convolution Test')
    parser.add_argument('--is-inverse', '-inv', action='store_true',
            help='If set, perform inverse convolution',
            default=False)
    parser.add_argument('--input-size', '-sz', type=int, metavar='INT',
            help='Size of the input array for the convolution',
            default=50)
    parser.add_argument('--strides', '-st', nargs='+', type=int, metavar='INT',
            help='Stride for the convolution.  If --is-inverse, this is'
            ' the input stride', default=1)
    parser.add_argument('--dilations', '-di', nargs='+', type=int, metavar='INT',
            help='Dilation for the filter', default=[0])
    parser.add_argument('--filters', '-f', nargs='+', type=str, metavar='STR',
            help='comma-separated list of positive integers, '
            'to be used as the convolution filter')
    parser.add_argument('--paddings', '-p', nargs='+', type=int, metavar='INT',
            help='padding for both left and right')
    parser.add_argument('--max-input-val', '-m', type=int, metavar='INT',
            help='maximum value in input cells')

    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()
    input = np.random.randint(1, args.max_input_val, args.input_size)
    np.set_printoptions(linewidth=250)

    sdpi = itertools.product(args.filters, args.strides, args.dilations, args.paddings, [True, False])
    
    for fil, st, dil, pad, inv in sdpi: 
        filt = [int(i) for i in fil.split(',')]
        fri = (len(filt) - 1) // 2

        # calculate phase of the first valid position of the filter after pad
        fri_dilated = dilate_index(fri, dil)
        used_pad = min(pad, fri_dilated)
        phase = (fri_dilated - used_pad) % st 

        ct = ConvType(filt, fri, st, inv, phase, dil, pad, pad)
        tc = torch_conv(ct, input)
        i, p, mc_raw, mask = matrix_conv(ct, input)
        mc = mc_raw[mask == VALID_VAL]
        same = np.all(tc == mc)

        if tc.shape != mc.shape:
            print('Error: tc: {}, mc: {}'.format(tc.shape, mc.shape))
        if not same:
            print('Not the same:')
            print('in: ', input)
            print('tc: ', tc)
            print('mc: ', mc)

        print('inverse: {:5}, fil: {}, il: {}, s: {}, d: {}, p: {}, ph: {}, '
        'Same? {}'.format(inv, str(ct.filter(True)), args.input_size, st,
            dil, pad, phase, same))


