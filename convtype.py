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

import numpy as np

VALID_VAL = 0 
SKIP_VAL = -1
INVALID_VAL = -2

def remove_vals(x, *vals):
    for v in vals:
        x = x[x != v]
    return x


def dilate_index(i, dilation):
    return i * (dilation)


def dilate_array(x, dilation, fill_value=0):
    d = np.full([dilate_index(len(x) - 1, dilation) + 1],
            fill_value=fill_value)
    for i,v in enumerate(x):
        dp = dilate_index(i, dilation)
        d[dp] = v 
        
    return d


class ConvType(object):

    def __init__(self, filt, filt_ctr, stride, is_inverse, phase, dilation, lpad, rpad):

        self.filt = np.array(filt, dtype=np.float)
        self.filt_ctr = filt_ctr 
        self.stride = stride
        self.is_inverse = is_inverse
        self.dilation = dilation
        self.phase = phase
        self._lpad = lpad
        self._rpad = rpad

    def usable_padding(self):
        '''check whether the dilated convolution filter could use up all
        left and right padding when traversing the input positions with
        the ref element'''
        fri = self.filter_ref_index(do_dilate=True)
        f_sz = self.filter_size(do_dilate=True)
        return fri >= self._lpad and (f_sz - 1 - fri) >= self._rpad
    
    def bad_input(self):
        return (self.stride < 1
        or self.phase > self.stride 
        or self.dilation < 1
        or self.filt_ctr < 0
        or self.filt_ctr >= len(self.filt)
        or self._lpad < 0
        or self._rpad < 0)
        
    def padding_type(self):
        '''return VALID, SAME, or CUSTOM based on the filter size
        and paddings requested'''
        fri = self.filter_ref_index(do_dilate=True)
        f_sz = self.filter_size(do_dilate=True)
        fri_rev = f_sz - 1 - fri

        if self._lpad == self._rpad == 0: return 'VALID'
        elif self._lpad == fri and self._rpad == fri_rev: return 'SAME'
        else: return 'CUSTOM'

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
            return dilate_index(self.filt_ctr, self.dilation)
        else:
            return self.filt_ctr

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

        fc = self.filter_ref_index(do_dilate=True)
        fci = self.filter_size(do_dilate=True) - fc 
        filt = self.filter(do_dilate=True)

        # 
        if self.is_inverse:
            filt = np.flip(filt, 0)
            #fc, fci = fci-1, fc


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

        # print(mat)
        return mat, mask

    def conv(self, input):
        # apply stride to the input rather than the output when doing the
        # inverse.
        if self.is_inverse:
            processed = dilate_array(input, self.stride, 0)
        else:
            processed = input 

        mat, mask = self.conv_mat(len(processed))
        conv = np.matmul(mat, processed)

        return input, processed, conv, mask 

