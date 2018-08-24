import tensorflow as tf
import convtype as ctyp
import numpy as np
from tensorflow.nn import convolution
from tensorflow.contrib.nn import conv1d_transpose
from sys import stderr


def wrap_input(x):
    return tf.constant(np.expand_dims(np.expand_dims(x, 0), 2),
            dtype=tf.float64)

def wrap_filt(x):
    return tf.constant(np.expand_dims(np.expand_dims(x, 1), 2),
            dtype=tf.float64)

def un_wrap(x):
    # only squeeze B and C, leaving W intact.
    return tf.squeeze(x, [0,2]).numpy()

def test(wanted_input_sz, filter_sz, stride, padding, dilation, max_val):

    assert tf.executing_eagerly()

    filter = np.random.randint(1, max_val, filter_sz)
    dilated_filter = ctyp.dilate_array(filter, dilation)
    dilated_filter_sz = len(dilated_filter)
    mask = ctyp.conv_mask(wanted_input_sz, dilated_filter_sz, stride, padding)
    matrix_sz = mask.shape[0]

    center_index = (dilated_filter_sz - 1) // 2
    mat = ctyp.make_mat(matrix_sz, dilated_filter, center_index)

    input = np.random.randint(1, max_val, matrix_sz)
    mm_conv = ctyp.do_mask(np.matmul(mat, input), mask)
    try:
        tf_conv = un_wrap(convolution(wrap_input(input), wrap_filt(filter),
            padding, [stride], [dilation]))
    except:
        tf_conv = np.array([])

    #print(mat)
    #print(mask)

    mm_convt = np.matmul(np.transpose(mat, (1, 0)), ctyp.un_mask(mm_conv, mask == 0))

    output_shape = tf.constant([1, matrix_sz, 1]) 
    
    try:
        tf_convt = un_wrap(conv1d_transpose(wrap_input(tf_conv),
            wrap_filt(dilated_filter), output_shape, stride, padding))
    except:
        tf_convt = np.array([])

    passed = tf_conv.shape == mm_conv.shape \
            and all(tf_conv == mm_conv) \
            and tf_convt.shape == mm_convt.shape \
            and all(tf_convt == mm_convt)
    return passed, (mm_conv.astype('f'), tf_conv, mm_convt.astype('f'), tf_convt)


def conv(input, matrix_sz, filt, inv, st, pad, dil):
    '''implement the equivalent TensorFlow convolution or deconvolution based
    on the parameters in conv_type'''
    assert tf.executing_eagerly()
    
    # B x W x C
    input_ten = wrap_input(input)
    filt_ten = wrap_filt(filt)

    strides = [st]
    dilation = [dil]

    with tf.device('/cpu:0'):
        try:
            if inv:
                output_shape = tf.constant([1, matrix_sz, 1])
                conv_ten = conv1d_transpose(input_ten, filt_ten, output_shape, st, pad)
                cmd = 'tf.contrib.nn.conv1d_transpose(input, filter, {}, {}, {})'.format(
                                str(output_shape.numpy()), st, pad)
            else:
                conv_ten = convolution(input_ten, filt_ten, pad, strides, dilation) 
                cmd = 'tf.nn.convolution(input, filter, {}, {}, {})'.format(
                        pad, str(strides), str(dilation))
        except (TypeError, tf.errors.InvalidArgumentError, ValueError) as te:
            return np.array([]), 'Exception: ' + str(te) 

        conv = un_wrap(conv_ten)
    return conv, cmd

