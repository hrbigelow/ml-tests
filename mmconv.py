import numpy as np

def dilate_array(x, dilation):
    '''e.g.
    x = [1,2,3], dilation = 2
    returns: [1, 0, 0, 2, 0, 0, 3]
    works with multiple dimension arrays as well
    '''
    assert len(x.shape) == len(dilation)
    for a, s in enumerate(x.shape):
        locs = np.repeat(range(1, x.shape[a]), dilation[a] - 1)
        x = np.insert(x, locs, 0, axis=a)
    return x


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


def get_padding(pad, input_sz, filter_sz, stride):
    '''Converts a padding strategy into actual padding values'''
    if isinstance(pad, tuple):
        return pad
    elif isinstance(pad, str):
        # use tensorflow-style padding calculation
        if pad == 'VALID':
            return 0, 0
        elif pad == 'SAME':
            output_sz = (input_sz + stride - 1) // stride
            padding_needed = max(0, (output_sz - 1) * stride +
                    filter_sz - input_sz)
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            return left_pad, right_pad
        else:
            raise ValueError
    elif isinstance(pad, int):
        return pad, pad


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

# TensorFlow's VALID padding scheme puts more padding on the right if the total
# padding needed is odd-lengthed.  This choice corresponds with a center_index
# that is left-of-center.
def center_index(filter_sz):
    s = 1
    if isinstance(filter_sz, list):
        return [(f - s) // 2 for f in filter_sz]
    else:
        return (filter_sz - s) // 2


def make_mask(input_sz, filter_sz, stride, padding_code, api_mode):
    '''produce the implicitly used mask corresponding to these settings for
    torch or tensorflow calls.  '''
    left_wing_sz = center_index(filter_sz)
    right_wing_sz = filter_sz - left_wing_sz - 1
    lpad, rpad = get_padding(padding_code, input_sz, filter_sz, stride)
    left_invalid = max(left_wing_sz - lpad, 0)
    right_invalid = max(right_wing_sz - rpad, 0)
    mid_sz = input_sz - left_invalid - right_invalid
    partial_stride = (mid_sz - 1) % stride
    mid_sz -= partial_stride

    mask = [False] * left_invalid
    for i in range(mid_sz):
        if i % stride == 0: mask += [True]
        else: mask += [False]
    if api_mode == 'Torch':
        mask += [False] * partial_stride
    elif api_mode == 'TensorFlow':
        mask += [False] * partial_stride

    mask += [False] * right_invalid

    return np.array(mask), partial_stride

def mask_repr(mask):
    return ''.join(list(map(lambda x: 'T' if x else '_', mask)))

def filter_repr(filt):
    return ''.join(list(map(lambda x: '*' if x else '-', filt)))

def array_equal(a, b):
    return a.shape == b.shape and (a == b).all()

def conv(input, matrix, mask):
    mm_conv = do_mask(np.matmul(matrix, input), mask)
    mm_convt = np.matmul(np.transpose(matrix, (1, 0)), un_mask(mm_conv, mask))
    return mm_conv, mm_convt

