import numpy as np
import torch
import tensorflow as tf
from torch.nn import functional as F
import mmconv as mmc
import fold

tf.enable_eager_execution()


def prepare_inputs(matrix_sz, filter_sz, stride, padding, dilation, max_val):
    filter = np.random.randint(1, max_val, filter_sz)
    dilated_filter = mmc.dilate_array(filter, dilation)
    dilated_filter_sz = len(dilated_filter)
    mask, partial_stride = mmc.make_mask(matrix_sz, dilated_filter_sz, stride, padding)

    assert len(mask) == matrix_sz
    matrix = mmc.make_matrix(matrix_sz, dilated_filter)

    input = np.random.randint(1, max_val, matrix_sz)
    return input, filter, dilated_filter, matrix, mask, partial_stride


def prepare_inputs2(matrix_sz, filter_sz, stride, padding, dilation, max_val):
    input = np.random.randint(1, max_val, matrix_sz)
    filter = np.random.randint(1, max_val, filter_sz)
    dilated_filter = mmc.dilate_array(filter, dilation)
    dilated_filter_sz = len(dilated_filter)
    fo = fold.Fold(matrix_sz, dilated_filter_sz, stride, padding, dilation)
    return input, filter, dilated_filter, fo 


# Torch
def torch_do_wrap(x):
    return torch.tensor(np.expand_dims(np.expand_dims(x, 0), 0),
            dtype=torch.float64)

def torch_un_wrap(x):
    return np.squeeze(np.squeeze(x.numpy(), 0), 0)

def torch_do_conv(input, filter, stride, padding, output_padding, dilation):
    n_spatial_dim = len(input.shape)  
    convs = [F.conv1d, F.conv2d, F.conv3d]
    tconvs = [F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d]

    th_conv = torch_un_wrap(convs[n_spatial_dim - 1](torch_do_wrap(input),
        torch_do_wrap(filter),
        None, stride, padding, dilation, 1))

    groups = 1
    th_convt = torch_un_wrap(tconvs[n_spatial_dim - 1](torch_do_wrap(th_conv),
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
    n_spatial_dim = len(input.shape)
    assert tf.executing_eagerly()
    try:
        tf_conv = tf_un_wrap(tf.nn.convolution(tf_wrap_input(input), tf_wrap_filt(filter),
            padding, [stride], [dilation]))
    except:
        tf_conv = np.array([])

    output_shape = tf.constant([1, len(input), 1]) 
    dilated_filter = mmc.dilate_array(filter, dilation)
    convts = [
            tf.contrib.nn.conv1d_transpose,
            tf.nn.conv2d_transpose,
            tf.nn.conv3d_transpose]
    try:
        tf_convt = tf_un_wrap(convts[n_spatial_dim - 1](tf_wrap_input(tf_conv),
            tf_wrap_filt(dilated_filter), output_shape, stride, padding))
    except:
        tf_convt = np.empty([0] * n_spatial_dim)
    return tf_conv, tf_convt




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

    print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format('API', 'FILT_SZ',
        'STRIDE', 'PAD', 'DIL', 'CONV_SZ', 'TCONV_SZ', 'CONV_EQ', 'TCONV_EQ', 'MASK', 'FILT'))

    for f_sz in range(1, args.filter_size_max + 1):
        for st in range(1, args.stride_max + 1):
            for pad in range((f_sz - 1) // 2 + 1):
                for dil in range(1, args.dilation_max + 1):
                # for dil in range(1, 1):
                    input, filter, dilated_filter, matrix, mask, out_pad = \
                            prepare_inputs(args.input_size, f_sz, st, pad, dil, max_val)
                    input, filter, dilated_filter, fo = \
                            prepare_inputs(args.input_size, f_sz, st, pad, dil, max_val)
                    mm_conv, mm_convt = fo.conv(input) 
                    mm_conv, mm_convt = mmc.conv(input, matrix, mask)
                    torch_conv, torch_convt = torch_do_conv(input, filter, st, pad, out_pad, dil)
                    eq = array_equal(mm_conv, torch_conv)
                    teq = array_equal(mm_convt, torch_convt)
                    conv_sz = len(mm_conv)
                    convt_sz = len(mm_convt)
                    print('Torch\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(f_sz,
                        st, pad, dil, conv_sz, convt_sz, eq, teq, 
                        mmc.mask_repr(mask), mmc.filter_repr(dilated_filter)))

    for f_sz in range(1, args.filter_size_max + 1):
        for st in range(1, args.stride_max + 1):
            for pad in ('SAME', 'VALID'):
                # TensorFlow doesn't support dilations > 1 if stride > 1
                max_dil = 1 if st > 1 else args.dilation_max
                for dil in range(1, max_dil + 1):
                    input, filter, dilated_filter, matrix, mask, out_pad = \
                            prepare_inputs(args.input_size, f_sz, st, pad, dil, max_val)
                    mm_conv, mm_convt = mmc.conv(input, matrix, mask)
                    tf_conv, tf_convt = tf_do_conv(input, filter, st, pad, dil)
                    eq = array_equal(mm_conv, tf_conv)
                    teq = array_equal(mm_convt, tf_convt)
                    conv_sz = len(mm_conv)
                    convt_sz = len(mm_convt)
                    print('TensorFlow\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(f_sz,
                        st, pad, dil, conv_sz, convt_sz, eq, teq, 
                        mmc.mask_repr(mask), mmc.filter_repr(dilated_filter)))


