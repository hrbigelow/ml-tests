import tensorflow as tf
import convtype as ctyp
import numpy as np
from tensorflow.nn import convolution
from tensorflow.contrib.nn import conv1d_transpose
from sys import stderr

def get_ct(matrix_sz, filt, stride, padding_type, dilation):
    '''Produce a ConvType object that corresponds with these settings'''
    filt_ctr = len(filt) // 2
    phase = 'LEFTMAX'
    return ctyp.ConvType(matrix_sz, filt, filt_ctr, stride, False,
            phase, dilation, padding_type)

def get_ct_transpose(matrix_sz, filt, stride, padding_type, dilation):
    '''Produce a ConvType object that corresponds with these settings'''
    filt_ctr = len(filt) // 2
    phase = 'LEFTMAX'
    return ctyp.ConvType(matrix_sz, filt, filt_ctr,
            stride, True, phase, dilation, padding_type)

def conv(input, filt, inv, st, pad, dil):
    '''implement the equivalent TensorFlow convolution or deconvolution based
    on the parameters in conv_type'''
    def add_dims(ten, *dims):
        for d in dims:
            ten = tf.expand_dims(ten, d)
        return ten

    ct = conv_type
    assert tf.executing_eagerly()
    
    # B x W x C
    input_ten = add_dims(tf.constant(input, dtype=tf.float64), 0, 2)
    # print('input_ten.shape = ', input_ten.shape)

    filt_ten = add_dims(tf.constant(ct.filter(do_dilate=False), dtype=tf.float64), 1, 2) 
    # print('filt_ten.shape = ', filt_ten.shape)

    strides = [st]
    dilation = [dil]

    if pad not in ('VALID', 'SAME'):
        print('tf_conv: cannot execute with {} padding'.format(pad), file=stderr)
        return (np.array([]), 'Not executed')

    if ct.is_inverse:
        output_shape = tf.constant([1, ct.matrix_sz, 1])
        # conv_ten = tf.constant([[1, 2]])
        print(cmd_string)
        conv_ten = conv1d_transpose(input_ten, filt_ten, output_shape, st, pad)
        cmd = 'tf.contrib.nn.conv1d_transpose(value=input, filter=filter, ' \
                'output_shape={}, stride={}, padding={})'.format(
                        str(output_shape.numpy()), st, pad)
    else:
        conv_ten = convolution(input_ten, filt_ten, pad, st, dil) 
        cmd = 'tf.nn.convolution(input=input, filter=filter, ' \
                'padding={}, strides={}, dilation_rate={})'.format(
                pad, st, dil)

    conv = tf.squeeze(conv_ten).numpy()
    return conv, cmd

