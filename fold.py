# Utilities for multidimensional matrix and mask construction
import numpy as np
import mmconv as mmc

def pad(ary, dim, cnt, val):
    '''append ary along dim with cnt copies of val'''
    assert isinstance(ary, np.ndarray)
    shape = list(ary.shape)
    shape[dim] = cnt
    extra = np.full(shape, val)
    return np.concatenate([ary, extra], axis=dim)

def linidx(shape, index):
    '''calculate the linearized index from a multidimensional
    array of shape.
    For example:
    shape = [2,5,4,6,7]
    index = [1,2,1,3,4]
    returns: 4 + 3 * 7 + 1 * 6 * 7 + 2 * 4 * 6 * 7 + 1 * 5 * 4 * 6 * 7 =
             4 + 7 * (3 + 6 * (1 + 4 * (2 + 5 * (1)))) '''
    assert len(shape) == len(index)
    if len(index) == 1:
        return index[0]
    return index[-1] + shape[-1] * linidx(shape[:-1], index[:-1])


class Fold(object):
    '''implement folding and unfolding between one and multiple
    dimensions'''

    def __init__(self, input_sz, filter, stride, padding, api_mode):
        self.ndim = len(input_sz)
        assert len(filter.shape) == self.ndim
        assert len(stride) == self.ndim
        assert len(padding) == self.ndim

        self.filter = filter
        self.filter_sz = list(filter.shape)
        self.input_sz = input_sz
        self.stride = stride
        self.pad = padding
        self.mask = []
        self.partial_stride = []
        self.api_mode = api_mode

        for d in range(self.ndim):
            m, p = mmc.make_mask(self.input_sz[d], self.filter_sz[d],
                    self.stride[d], self.pad[d], self.api_mode)
            self.mask.append(m)
            self.partial_stride.append(p)


    def input_spacer_mask(self):
        m = np.full(self.input_sz, True)
        for d in reversed(range(1, self.ndim)):
            m = pad(m, d, self.filter_sz[d] - 1, False)
        end_idx = [s - 1 for s in self.input_sz]
        end = linidx(list(m.shape), end_idx)
        return m.reshape(-1)[:end + 1]


    def _unfold_filter(self):
        m = self.filter
        for d in reversed(range(1, self.ndim)):
            m = pad(m, d, self.input_sz[d] - 1, 0)
        end_idx = [s - 1 for s in self.filter_sz]
        key_idx = mmc.center_index(self.filter_sz)
        end = linidx(list(m.shape), end_idx) 
        key = linidx(list(m.shape), key_idx)
        return m.reshape(-1)[:end + 1], key


    def make_mask(self):
        v = np.array([True])
        for m in self.mask:
            v = np.concatenate(list(map(lambda b: v & b, m)))
        return v

    def conv_size(self):
        return list(map(lambda m: len(np.where(m)[0]), self.mask))


    def make_matrix(self):
        filter_vals, ki = self._unfold_filter()
        filter_vals = filter_vals.tolist()
        tmp_filter_sz = len(filter_vals)
        is_mask = self.input_spacer_mask()
        tmp_matrix_sz = len(is_mask)
        loff = tmp_matrix_sz - ki - 1
        roff = tmp_matrix_sz - tmp_filter_sz + ki

        lzero, ltrim = max(loff, 0), max(-loff, 0)
        rzero, rtrim = max(roff, 0), max(-roff, 0)

        values = [0] * lzero + filter_vals[ltrim:rtrim if rtrim != 0 else None] + [0] * rzero
        assert len(values) == tmp_matrix_sz * 2 - 1

        cells = []
        for i in reversed(range(tmp_matrix_sz)):
            cells += values[i:i + tmp_matrix_sz]

        tmp_mat = np.array(cells).reshape(tmp_matrix_sz, tmp_matrix_sz)
        mat = tmp_mat[is_mask,:][:,is_mask]
        return mat


    def conv(self, input):
        matrix = self.make_matrix()
        mask = self.make_mask()
        input_flat = input.reshape(-1)
        mm_conv_flat = mmc.do_mask(np.matmul(matrix, input_flat), mask)
        mm_conv = mm_conv_flat.reshape(self.conv_size())
        mm_convt_flat = np.matmul(np.transpose(matrix, (1, 0)), mmc.un_mask(mm_conv_flat, mask))  
        mm_convt = mm_convt_flat.reshape(self.input_sz)
        return mm_conv, mm_convt


