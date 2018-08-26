import numpy as np

def dilate_array(x, dilation):
    '''e.g.
    x = [1,2,3], dilation = 2
    returns: [1, 0, 0, 2, 0, 0, 3]
    '''
    dm = dilation - 1
    d = np.zeros([len(x) + (len(x) - 1) * dm])
    i = 0
    for v in x:
        d[i] = v
        i += dilation 
    return d


def un_mask(x, mask):
    '''Inserts zero-padding where mask is False.  For example:
    x = [1,2,3], mask = [True, False, False, True, False, True, False]
    returns: [1, 0, 0, 2, 0, 3, 0]
    '''
    it = iter(x)
    try:
        u = np.array([next(it) if m else 0 for m in mask])
        return u
    except StopIteration:
        print('Error: not enough input values: ', x, mask)
        raise
    
def do_mask(x, mask):
    '''Remove elements of x where mask is False'''  
    return x[mask] 


def get_padding(pad, filter_sz, center_index):
    '''Converts a padding strategy into actual padding values'''
    if isinstance(pad, tuple):
        return pad
    elif isinstance(pad, str):
        if pad == 'VALID':
            return 0, 0
        elif pad == 'SAME':
            return center_index, filter_sz - center_index - 1
        else:
            raise ValueError
    elif isinstance(pad, int):
        return pad, pad

def center_index(filter_sz):
    return (filter_sz - 1) // 2


def make_matrix(matrix_sz, filter):
    ''' outputs: mat (input_sz x input_sz)
        use as: conv_raw = np.matmul(mat, input) 
    '''
    filter = filter.tolist()
    filter_sz = len(filter)
    c = center_index(filter_sz)
    left_zeros = matrix_sz - c - 1
    right_zeros = matrix_sz - filter_sz + c
    values = [0] * left_zeros + filter + [0] * right_zeros 

    cells = []
    for i in reversed(range(matrix_sz)):
        cells += values[i:i + matrix_sz]

    return np.array(cells).reshape(matrix_sz, matrix_sz)


def make_mask(input_sz, filter_sz, stride, padding_code):
    '''produce the implicitly used mask corresponding to these settings for
    torch or tensorflow calls.
    Returns:
        mask
        snip: the number of '''
    lw = center_index(filter_sz)
    rw = filter_sz - lw - 1
    lpad, rpad = get_padding(padding_code, filter_sz, lw)

    left_invalid = max(lw - lpad, 0)
    right_invalid = max(rw - rpad, 0)
    mid_sz = input_sz - left_invalid - right_invalid
    partial_stride = (mid_sz - 1) % stride
    mid_sz -= partial_stride

    mask = [False] * left_invalid
    for i in range(mid_sz):
        if i % stride == 0: mask += [True]
        else: mask += [False]
    mask += [False] * (right_invalid + partial_stride)

    return np.array(mask), partial_stride

def mask_repr(mask):
    return ''.join(list(map(lambda x: 'T' if x else '_', mask)))

def filter_repr(filt):
    return ''.join(list(map(lambda x: '*' if x else '-', filt)))

def conv(input, matrix, mask):
    mm_conv = do_mask(np.matmul(matrix, input), mask)
    mm_convt = np.matmul(np.transpose(matrix, (1, 0)), un_mask(mm_conv, mask))
    return mm_conv, mm_convt

