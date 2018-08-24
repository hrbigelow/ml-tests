import numpy as np
import torch
import tensorflow as tf

tf.enable_eager_execution()

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


def make_mask(input_sz, filter_sz, center_index, stride, padding_code):
    '''produce the implicitly used mask corresponding to these settings for
    torch or tensorflow calls'''
    lw = center_index
    rw = filter_sz - lw - 1
    lpad, rpad = get_padding(padding_code, filter_sz, lw)

    left_invalid = max(lw - lpad, 0)
    right_invalid = max(rw - rpad, 0)
    mid_sz = input_sz - left_invalid - right_invalid
    snip = (mid_sz - 1) % stride
    mid_sz -= snip

    mask = [False] * left_invalid
    mid_sz
    for i in range(mid_sz):
        if i % stride == 0: mask += [True]
        else: mask += [False]
    mask += [False] * right_invalid

    return np.array(mask)



def prepare_inputs(wanted_input_sz, filter_sz, stride, padding, dilation, max_val):
    filter = np.random.randint(1, max_val, filter_sz)
    dilated_filter = dilate_array(filter, dilation)
    dilated_filter_sz = len(dilated_filter)
    center_index = (dilated_filter_sz - 1) // 2
    mask = make_mask(wanted_input_sz, dilated_filter_sz, center_index, stride, padding)
    matrix_sz = mask.shape[0]
    matrix = make_mat(matrix_sz, dilated_filter, center_index)

    input = np.random.randint(1, max_val, matrix_sz)
    return input, filter, dilated_filter, matrix, mask


def matmul_conv(input, matrix, mask):
    mm_conv = do_mask(np.matmul(matrix, input), mask)
    mm_convt = np.matmul(np.transpose(matrix, (1, 0)), un_mask(mm_conv, mask))
    return mm_conv, mm_convt


# Torch
def torch_do_wrap(x):
    return torch.tensor(np.expand_dims(np.expand_dims(x, 0), 0),
            dtype=torch.float64)

def torch_un_wrap(x):
    return np.squeeze(np.squeeze(x.numpy(), 0), 0)

def torch_do_conv(input, filter, stride, padding, dilation):
    th_conv = torch_un_wrap(torch.nn.functional.conv1d(torch_do_wrap(input),
        torch_do_wrap(filter),
        None, stride, padding, dilation, 1))

    output_padding = 0
    groups = 1
    th_convt = torch_un_wrap(torch.nn.functional.conv_transpose1d(torch_do_wrap(th_conv),
        torch_do_wrap(filter), None,
        stride, padding, output_padding, groups, dilation))
    return th_conv, th_convt


# TensorFlow
def tf_wrap_input(x):
    return tf.constant(np.expand_dims(np.expand_dims(x, 0), 2),
            dtype=tf.float64)

def tf_wrap_filt(x):
    return tf.constant(np.expand_dims(np.expand_dims(x, 1), 2),
            dtype=tf.float64)

def tf_un_wrap(x):
    # only squeeze B and C, leaving W intact.
    return tf.squeeze(x, [0,2]).numpy()


def tf_do_conv(input, filter, stride, padding, dilation):
    assert tf.executing_eagerly()
    try:
        tf_conv = tf_un_wrap(tf.nn.convolution(tf_wrap_input(input), tf_wrap_filt(filter),
            padding, [stride], [dilation]))
    except:
        tf_conv = np.array([])

    output_shape = tf.constant([1, len(input), 1]) 
    dilated_filter = dilate_array(filter, dilation)
    try:
        tf_convt = tf_un_wrap(tf.contrib.nn.conv1d_transpose(tf_wrap_input(tf_conv),
            tf_wrap_filt(dilated_filter), output_shape, stride, padding))
    except:
        tf_convt = np.array([])
    return tf_conv, tf_convt


def array_equal(a, b):
    return a.shape == b.shape and all(a == b)

    passed = tf_conv.shape == mm_conv.shape \
            and all(tf_conv == mm_conv) \
            and tf_convt.shape == mm_convt.shape \
            and all(tf_convt == mm_convt)
    return passed, (mm_conv.astype('f'), tf_conv, mm_convt.astype('f'), tf_convt)

def mask_repr(mask):
    return ''.join(list(map(lambda x: 'T' if x else '_', mask)))

def filter_repr(filt):
    return ''.join(list(map(lambda x: '*' if x else '-', filt)))


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Convolution Test')
    parser.add_argument('--input-size', '-sz', type=int, metavar='INT',
            help='Size of the input to use for the convolution',
            default=100)
    parser.add_argument('--max-val', '-max', type=int, metavar='INT',
            help='Maximum value to populate input and filter elements',
            default=100)
    parser.add_argument('--filter-size-max', '-fil', type=int, metavar='INT',
            help='Maximum size of filter to test')
    parser.add_argument('--stride-max', '-str', type=int, metavar='INT',
            help='Maximum stride to test')
    parser.add_argument('--dilation-max', '-dil', type=int, metavar='INT',
            help='Maximum dilation to test')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    max_val = args.max_val 
    np.set_printoptions(linewidth=158)

    for f_sz in range(1, args.filter_size_max + 1):
        for st in range(1, args.stride_max + 1):
            for pad in range((f_sz - 1) // 2 + 1):
                for dil in range(1, args.dilation_max + 1):
                # for dil in range(1, 1):
                    input, filter, dilated_filter, matrix, mask = \
                            prepare_inputs(args.input_size, f_sz, st, pad, dil, max_val)
                    mm_conv, mm_convt = matmul_conv(input, matrix, mask)
                    torch_conv, torch_convt = torch_do_conv(input, filter, st, pad, dil)
                    eq = array_equal(mm_conv, torch_conv)
                    teq = array_equal(mm_convt, torch_convt)
                    print('Torch\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(f_sz,
                        st, pad, dil, eq, teq, mask_repr(mask), filter_repr(dilated_filter)))

    for f_sz in range(1, args.filter_size_max + 1):
        for st in range(1, args.stride_max + 1):
            for pad in ('SAME', 'VALID'):
                # TensorFlow doesn't support dilations > 1 if stride > 1
                max_dil = 1 if st > 1 else args.dilation_max
                for dil in range(1, max_dil + 1):
                    input, filter, dilated_filter, matrix, mask = \
                            prepare_inputs(args.input_size, f_sz, st, pad, dil, max_val)
                    mm_conv, mm_convt = matmul_conv(input, matrix, mask)
                    tf_conv, tf_convt = tf_do_conv(input, filter, st, pad, dil)
                    eq = array_equal(mm_conv, tf_conv)
                    teq = array_equal(mm_convt, tf_convt)
                    print('TensorFlow\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(f_sz,
                        st, pad, dil, eq, teq, mask_repr(mask), filter_repr(dilated_filter)))


