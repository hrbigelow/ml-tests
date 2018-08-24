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


def un_mask(x, mask):
    '''if x = a[mask], produce a, where a[np.logical_not(mask)] = 0'''
    it = iter(x)
    try:
        u = np.array([next(it) if m else 0 for m in mask])
        return u
    except StopIteration:
        print('Error: not enough input values: ', x, mask)
        raise
    
def do_mask(x, mask):
    return x[mask == 0] 


def make_mat(matrix_sz, filt, center_index):
    ''' outputs: mat (input_sz x input_sz)
        use as: conv_raw = np.matmul(mat, input) 
    '''
    c = center_index
    filt = filt.tolist()
    filt_sz = len(filt)
    left_zeros = matrix_sz - c - 1
    right_zeros = matrix_sz - filt_sz + c
    values = [0] * left_zeros + filt + [0] * right_zeros 

    mat = []
    for i in reversed(range(matrix_sz)):
        mat += values[i:i + matrix_sz]

    return np.array(mat).reshape(matrix_sz, matrix_sz)


def get_padding(padding, filter_sz, center_index):
    if isinstance(padding, tuple):
        return padding
    elif isinstance(padding, str):
        if padding == 'VALID':
            return 0, 0
        elif padding == 'SAME':
            return center_index, filter_sz - center_index - 1
        else:
            raise ValueError
    elif isinstance(padding, int):
        return padding, padding


def conv_mask(input_sz, filter_sz, stride, padding_code):
    '''produce the implicitly used mask corresponding to these settings for
    F.conv1d call'''
    lw, rw = ((filter_sz - 1) // 2), (filter_sz // 2)
    lpad, rpad = get_padding(padding_code, filter_sz, lw)

    left_invalid = max(lw - lpad, 0)
    right_invalid = max(rw - rpad, 0)
    mid_sz = input_sz - left_invalid - right_invalid
    snip = (mid_sz - 1) % stride
    mid_sz -= snip

    mask = [-2] * left_invalid
    mid_sz
    for i in range(mid_sz):
        if i % stride == 0: mask += [0]
        else: mask += [-1]
    mask += [-2] * right_invalid

    return np.array(mask)


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
        self._lpad, self._rpad = \
                get_padding(padding, self.filter_size(), self.ref_index())


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
        filt = self.filter(do_dilate=True)
        mat = make_mat(self.matrix_sz, filt, self.ref_index())
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
            processed = un_mask(input, bool_mask)
            conv_raw = np.matmul(mat, processed)
            conv = conv_raw

        else:
            processed = input 
            conv_raw = np.matmul(mat, processed)
            conv = do_mask(conv_raw, bool_mask)

        return processed, conv_raw, conv, mask 

