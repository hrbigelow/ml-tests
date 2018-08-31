# test equivalence of mmc.make_matrix and fold.make_matrix, when the extra dimensions
# are size 1.
import fold
import mmconv as mmc
import numpy as np

def prod(x):
    p = 1
    for e in x:
        p *= e
    return p


filter_sz = [5, 1, 1]
input_sz = [45, 1, 1]
ndim = len(filter_sz)

filt = np.array(list(range(1, prod(filter_sz) + 1))).reshape(filter_sz)
input = np.array(list(range(1, prod(input_sz) + 1))).reshape(input_sz)

for s in range(1, 5):
    for p in ('SAME', 'VALID'):
        for d in range(1, 4):
            stride = [s] * ndim 
            pad = [p] * ndim
            dil = [d] * ndim
            fo = fold.Fold(input_sz, filter_sz, stride, pad)
            fmat = fo.make_matrix(filt)
            fmask = fo.make_mask()
            fextra = fo.partial_stride
            mmat = mmc.make_matrix(input_sz[0], np.squeeze(filt))
            mmask, mextra = mmc.make_mask(input_sz[0], filter_sz[0], stride[0], pad[0])
            mat_eq = mmc.array_equal(fmat, mmat)
            mask_eq = mmc.array_equal(fmask, mmask)
            print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(s, p, d, mat_eq,
                mask_eq, str(fextra), str(mextra)))


