# Utilities for multidimensional matrix and mask construction
import mmconv as mmc

def joinl(vals, sep):
    '''join a list of lists using sep_list'''
    if len(vals) == 0: 
        return None 
    j = vals[0]
    for l in vals[1:]:
        j += sep + l
    return j


def joinv(val, rep, sep):
    '''join rep repetitions of val using sep.
    works if val and sep are scalars or lists'''
    return (val + sep) * (rep - 1) + val 



class Fold(object):
    '''implement folding and unfolding between one and multiple
    dimensions'''

    def __init__(self, filter_dims, input_dims, stride, padding, dilation):
        assert len(filter_dims) == len(input_dims) 
        self._fd = filter_dims
        self._id = input_dims
        self.ndims = len(self._fd)

    def _i(self, d):
        '''input width in the d-th dimension (from 0)'''
        if d == 0:
            return self._id[d]
        else:
            sep = self._f(d - 1) - 1
            return joinv(self._i(d - 1), self._id[d], sep)


    def _f(self, d):
        '''filter width in the d-th dimension (from 0)'''
        if d == 0:
            return self._fd[d]
        else:
            sep = self._i(d - 1) - 1
            return joinv(self._f(d - 1), self._fd[d], sep)

    def _k(self, d):
        '''position of the key element in the d-th dimension (from 0)'''
        if d == 0:
            return self._f(d) // 2
        else:
            sep = self._i(d - 1) - 1
            return (self._f(d - 1) + sep) * (self._fd[d] // 2) + self._k(d - 1)


    def _ism(self, d):
        '''spacer mask for the input unfolding'''
        if d == 0:
            return [True] * self._i(d)
        else:
            sep = [False] * (self._f(d - 1) - 1)
            return joinv(self._ism(d - 1), self._id[d], sep)

    def _fsm(self, d):
        '''spacer mask for the filter unfolding'''
        if d == 0:
            return [True] * self._fd[d]
        else:
            sep = [False] * (self._i(d - 1) - 1) 
            return joinv(self._fsm(d - 1), self._fd[d], sep)


    def _vm(self, d):
        mask = mmc.make_mask(self._i(d), self._f(d), self.stride[d], self.pad[d])
        if d == 0:
            return mask
        else:
            sep = [False] * (self._f(d - 1) - 1)
            return joinl([[m and s for s in sub] for m in mask], sep) 



    def filter_sz(self):
        return self._f(self.ndims - 1)

    def input_sz(self):
        return self._i(self.ndims - 1)

    def key_ind(self):
        return self._k(self.ndims - 1)

    def input_spacer_mask(self):
        return self._ism(self.ndims - 1)

    def validity_mask(self):
        return self._vm(self.ndims - 1)

    def unfold_filter(self, filter):
        sz = 1
        for d in self._fd:
            sz *= d
        filter_vals = filter.reshape(sz) 
        filter_mask = self._fsm(self.ndims - 1) 
        return mmc.un_mask(filter_vals, filter_mask)

    def make_matrix(self, filter):
        ki = self.key_ind()
        tmp_matrix_sz = self.input_sz()
        tmp_filter_sz = self.filter_sz()
        filter_vals = self.unfold_filter(filter)
        left_zeros = tmp_matrix_sz - ki - 1
        right_zeros = tmp_matrix_sz - tmp_filter_sz + ki
        values = [0] * left_zeros + filter_vals + [0] * right_zeros

        cells = []
        for i in reversed(range(tmp_matrix_sz)):
            cells += values[i:i + tmp_matrix_sz]

        mat_tmp = np.array(cells).reshape(matrix_sz, matrix_sz)
        is_mask = self._ism(self.ndims - 1)
        mat = mat_tmp[is_mask,:][:,is_mask]
        return mat




