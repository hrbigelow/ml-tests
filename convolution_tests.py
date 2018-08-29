import numpy as np
import torch
import tensorflow as tf
from torch.nn import functional as F
import mmconv as mmc
import fold

tf.enable_eager_execution()


def prepare_inputs(matrix_sz, filter_sz, stride, padding, dilation, max_val, api_mode):
    input = np.random.randint(1, max_val, matrix_sz)
    filter = np.random.randint(1, max_val, filter_sz)
    dilated_filter = mmc.dilate_array(filter, dilation)
    fo = fold.Fold(matrix_sz, dilated_filter, stride, padding, api_mode)
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
    '''add a pseudo-dimension for batch and channel'''
    return tf.constant(np.expand_dims(np.expand_dims(x, 0), -1),
            dtype=tf.float64)

def tf_wrap_filt(x):
    '''add pseudo-dimensions for in_channel and out_channel'''
    return tf.constant(np.expand_dims(np.expand_dims(x, -1), -1),
            dtype=tf.float64)

def tf_un_wrap(x):
    '''remove batch and channel (first and last dimensions)'''
    # only squeeze B and C, leaving W intact.
    return tf.squeeze(x, [0,-1]).numpy()


def tf_do_conv(input, dilated_filter, conv_sz, stride, padding):
    n_spatial_dim = len(input.shape)
    assert tf.executing_eagerly()
    convs = [tf.nn.conv1d,
            tf.nn.conv2d,
            tf.nn.conv3d]
    if n_spatial_dim == 1:
        conv_stride = stride[0]
    else:
        conv_stride = [1] + stride + [1]
    try:
        with tf.device('/cpu:0'):
            tf_conv_raw = \
                    convs[n_spatial_dim - 1]( tf_wrap_input(input),
                    tf_wrap_filt(dilated_filter), conv_stride, padding)
            tf_conv = tf_un_wrap(tf_conv_raw)
        output_shape = tf_wrap_input(input).shape 
    except:
        tf_conv = np.array([])
        output_shape = [0, 0, 0]

    #output_shape = tf.constant([1] + conv_sz + [1]) 
    stride_fix = [1] + stride + [1]
    convts = [
            tf.contrib.nn.conv1d_transpose,
            tf.nn.conv2d_transpose,
            tf.nn.conv3d_transpose]
    try:
        with tf.device('/cpu:0'):
            tf_convt = tf_un_wrap(convts[n_spatial_dim - 1](tf_wrap_input(tf_conv),
                tf_wrap_filt(dilated_filter), output_shape, conv_stride, padding))
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
    parser.add_argument('--ndims', '-nd', type=int, metavar='INT',
            default=1, help='Number of dimensions')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    max_val = args.max_val 
    np.set_printoptions(linewidth=158)

    mask_hdr = '\t'.join(['MASK{}'.format(i) for i in range(args.ndims)])

    print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
        'LINE_NO', 'API-------', 'FILT_SZ',
        'STRIDE', 'PAD', 'DIL', 'CONV_SZ', 'TCONV_SZ', 'CONV_EQ',
        'TCONV_EQ', mask_hdr, 'FILT'))

    api_mode = 'Torch'
    if True:
    #if False:
        for f_sz in range(1, args.filter_size_max + 1):
            for st in range(1, args.stride_max + 1):
                for pad in range((f_sz - 1) // 2 + 1):
                    for dil in range(1, args.dilation_max + 1):
                    # for dil in range(1, 1):
                        i_szs = [args.input_size] * args.ndims
                        f_szs = [f_sz] * args.ndims
                        sts = [st] * args.ndims
                        pads = [pad] * args.ndims
                        dils = [dil] * args.ndims

                        # we ignore the dilated_filter returned.  torch handles
                        # dilation itself, and Fold stores its own copy.
                        input, filter, _, fo = \
                                prepare_inputs(i_szs, f_szs, sts, pads, dils, max_val, api_mode)
                        mm_conv, mm_convt = fo.conv(input) 
                        # mm_conv, mm_convt = mmc.conv(input, matrix, mask)
                        out_pad = fo.partial_stride
                        torch_conv, torch_convt = \
                                torch_do_conv(input, filter, sts, pads, out_pad, dils)
                        eq = mmc.array_equal(mm_conv, torch_conv)
                        teq = mmc.array_equal(mm_convt, torch_convt)
                        conv_sz = list(mm_conv.shape)
                        convt_sz = list(mm_convt.shape)
                        mask_fields = '\t'.join([mmc.mask_repr(m) for m in fo.mask])
                        print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                            api_mode, str(f_sz), str(sts), str(pads), str(dils),
                            str(conv_sz), str(convt_sz), eq, teq, 
                            mask_fields, mmc.filter_repr(fo.filter.reshape(-1))))


    api_mode = 'TensorFlow'
    row = 1
    for f_sz in range(1, args.filter_size_max + 1):
        for st in range(1, args.stride_max + 1):
            for pad in ('SAME', 'VALID'):
                # TensorFlow doesn't support dilations > 1 if stride > 1
                max_dil = 1 if st > 1 else args.dilation_max
                for dil in range(1, max_dil + 1):
                    i_szs = [args.input_size] * args.ndims
                    f_szs = [f_sz] * args.ndims
                    sts = [st] * args.ndims
                    pads = [pad] * args.ndims
                    dils = [dil] * args.ndims
                    # we ignore the filter returned here, since tf only uses
                    # dilated_filter, and fo stores the filter itself
                    input, _, dilated_filter, fo = prepare_inputs(
                            i_szs, f_szs, sts, pads, dils, max_val, api_mode)
                    mm_conv, mm_convt = fo.conv(input) 
                    conv_sz = fo.conv_size()
                    tf_conv, tf_convt = tf_do_conv(input, dilated_filter, conv_sz, sts, pad)
                    eq = mmc.array_equal(mm_conv, tf_conv)
                    teq = mmc.array_equal(mm_convt, tf_convt)
                    conv_sz = list(mm_conv.shape)
                    convt_sz = list(mm_convt.shape)
                    mask_fields = '\t'.join([mmc.mask_repr(m) for m in fo.mask])
                    print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                        row, api_mode, str(f_sz), str(sts), pad, str(dils), str(conv_sz),
                        str(convt_sz), eq, teq, mask_fields,
                        mmc.filter_repr(fo.filter.reshape(-1)[:30]) + ' ]'))
                    row += 1


