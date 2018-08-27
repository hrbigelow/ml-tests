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

    def __init__(self, input_sz, filter_sz, stride, padding, dilation):
        self.ndim = len(input_sz)
        assert len(filter_sz) == self.ndim
        assert len(stride) == self.ndim
        assert len(padding) == self.ndim
        assert len(dilation) == self.ndim

        self._fd = filter_sz
        self._id = input_sz
        self.stride = stride
        self.pad = padding
        self.dilation = dilation


    def _input_spacer_mask(self):
        m = np.full(self._id, True)
        for d in reversed(range(1, self.ndim)):
            m = pad(m, d, self._fd[d] - 1, False)
        end_idx = [s - 1 for s in self._id]
        end = linidx(list(m.shape), end_idx)
        return m.reshape(-1)[:end + 1]


    def _unfold_filter(self, filter):
        m = filter
        for d in reversed(range(1, self.ndim)):
            m = pad(m, d, self._id[d] - 1, 0)
        end_idx = [s - 1 for s in filter.shape]
        key_idx = [(s - 1) // 2 for s in self._fd]
        end = linidx(list(m.shape), end_idx) 
        key = linidx(list(m.shape), key_idx)
        return m.reshape(-1)[:end + 1], key


    def make_mask(self):
        v = np.array([True])
        partial_stride = []
        masked_sz = []
        for d in range(self.ndim):
            m, p = mmc.make_mask(self._id[d], self._fd[d], self.stride[d], self.pad[d])
            v = np.concatenate(list(map(lambda b: v & b, m)))
            partial_stride.append(p)
            masked_sz.append(len(np.where(m)[0]))
        return v, partial_stride, masked_sz


    def make_matrix(self, filter):
        filter_vals, ki = self._unfold_filter(filter)
        filter_vals = filter_vals.tolist()
        tmp_filter_sz = len(filter_vals)
        is_mask = self._input_spacer_mask()
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

