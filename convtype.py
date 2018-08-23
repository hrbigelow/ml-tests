# Experiments with convolutions using a proof-of-principle approach. 

# In these tests, a convolution is implemented as a single, square matrix
# multiplication of the input.  the main concepts of stride, filter width,
# dilation, fractional stride, left and right padding are all implemented
# by choosing 
# zeros.

# This approach is inspired by Naoki Shibuya's Medium article,
# https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0


import numpy as np

VALID_VAL = 0 
SKIP_VAL = -1
INVALID_VAL = -2


def dilate_index(i, dilation):
    return i * (dilation)


def dilate_array(x, dilation, fill_value=0):
    d = np.full([dilate_index(len(x) - 1, dilation) + 1],
            fill_value=fill_value)
    for i,v in enumerate(x):
        dp = dilate_index(i, dilation)
        d[dp] = v 
    return d


def unmask(x, mask):
    '''if x = a[mask], produce a, where a[np.logical_not(mask)] = 0'''
    it = iter(x)
    try:
        u = np.array([next(it) if m else 0 for m in mask])
        return u
    except StopIteration:
        print(x, mask)
        exit(1)
    

class ConvType(object):

    def __init__(self, matrix_sz, filt, filt_ctr, stride, is_inverse, phase, dilation, padding):

        self.matrix_sz = matrix_sz
        self.filt = np.array(filt, dtype=np.float)
        self.stride = stride
        self.is_inverse = is_inverse
        self.dilation = dilation
        self.set_filter_center(filt_ctr) 
        self.set_padding(padding)
        self.phase = phase


    def filter(self, do_dilate):
        if do_dilate:
            return dilate_array(self.filt, self.dilation)
        else:
            return self.filt

    def filter_size(self):
        return len(self.filter(True))

    def ref_index(self):
        return self.filt_ctr 

    def ref_index_rev(self):
        return self.filter_size() - 1 - self.ref_index()

    def set_filter_center(self, ftype):
        if isinstance(ftype, str):
            if ftype == 'LC':
                f_sz = self.filter_size()
                if self.is_inverse:
                    self.filt_ctr = f_sz // 2
                else:
                    self.filt_ctr = (f_sz - 1) // 2
        else:
            self.filt_ctr = ftype


    def set_padding(self, padding):
        if isinstance(padding, tuple):
            self._lpad, self._rpad = padding
        elif isinstance(padding, str):
            if padding == 'VALID':
                self._lpad = self._rpad = 0
            elif padding == 'SAME':
                self._lpad = self.ref_index()
                self._rpad = self.ref_index_rev()
            else:
                raise ValueError


    def bad_input(self):
        return (self.stride < 1
        or self.phase >= self.stride 
        or self.dilation < 1
        or self.filt_ctr < 0
        or self.filt_ctr >= len(self.filt)
        or self._lpad < 0
        or self._rpad < 0)
        

    def input_size(self):
        '''give the size of input that this ConvType will accept'''
        mask = self.mask()
        if self.is_inverse:
            return len(mask[mask == 0])
        else:
            return len(mask)
            

    def ref_bounds_avoid_pad(self):
        '''provide [beg, end) range where the filter reference element can
        be, not considering padding'''
        return (self.ref_index(), self.matrix_sz - self.ref_index_rev())


    def ref_bounds_allowed(self):
        '''provide [beg, end) range where the filter reference element can
        be, considering padding'''
        beg, end = self.ref_bounds_avoid_pad()
        return max(beg - self._lpad, 0), min(self.matrix_sz, end + self._rpad)

    def mask(self):
        '''generate a mask of valid output positions based on phase, stride,
        left padding, right padding, filter size, and filter reference position'''
        mask = np.full([self.matrix_sz], VALID_VAL)
        beg, end = self.ref_bounds_allowed()
        
        for i in range(self.matrix_sz):
            if i not in range(beg, end):
                mask[i] = INVALID_VAL
            elif i % self.stride != self.phase:
                mask[i] = SKIP_VAL
            else:
                mask[i] = VALID_VAL
        return mask


    def conv_mat(self):
        ''' outputs: mat (input_sz x input_sz)
            use as: conv_raw = np.matmul(mat, input) 
        '''
        c = self.ref_index()
        filt = self.filter(do_dilate=True).tolist()

        # construct the repeating pattern of elements, then reshape
        fl, fc, fr = filt[0:c], filt[c:c+1], filt[c+1:]
        m_sz = self.matrix_sz
        z = [0] * (m_sz - len(filt) + 1)
        values = (fc + fr + z + fl) * (m_sz - 1) + fc

        mat = np.array(values).reshape(m_sz, m_sz)
        if self.is_inverse:
            mat = np.transpose(mat, (1, 0))

        return mat


    def conv(self, input):
        # apply stride to the input rather than the output when doing the
        # inverse.
        mask = self.mask()
        mat = self.conv_mat()
        bool_mask = (mask == 0)

        if self.is_inverse:
            processed = unmask(input, bool_mask)
            conv_raw = np.matmul(mat, processed)
            conv = conv_raw

        else:
            processed = input 
            conv_raw = np.matmul(mat, processed)
            conv = conv_raw[bool_mask] 

        return processed, conv_raw, conv, mask 

