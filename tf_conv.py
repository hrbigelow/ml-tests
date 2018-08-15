import tensorflow as tf
import numpy as np
from tensorflow.nn import convolution
from tensorflow.contrib.nn import conv1d_transpose
from sys import stderr

def conv(conv_type, input):
    '''implement the equivalent TensorFlow convolution or deconvolution based
    on the parameters in conv_type'''
    def add_dims(ten, *dims):
        for d in dims:
            ten = tf.expand_dims(ten, d)
        return ten

    ct = conv_type
    assert tf.executing_eagerly()
    
    input_ten = add_dims(tf.constant(input, dtype=tf.float64), 0, 2)
    filt_ten = add_dims(tf.constant(ct.filter(do_dilate=False), dtype=tf.float64), 1, 2) 
    strides = [ct.stride]
    dilation = [ct.dilation]
    padding = ct.padding_type()
    if padding not in ('VALID', 'SAME'):
        print('tf_conv: cannot execute with {} padding'.format(padding), file=stderr)
        return (np.array([]), 'Not executed')

    if ct.is_inverse:
        if ct.dilation != 1:
            print('tf_conv: conv1d_transpose doesn\'t support dilation > 1', file=stderr)
            return (np.array([]), 'Not executed')
        output_shape = ()
        conv_ten = conv1d_transpose(input_ten, filt_ten, output_shape, ct.stride, padding)
        cmd_string = 'tf.contrib.nn.conv1d_transpose(value=input, filter=filter, ' \
                'output_shape=?, stride={}, padding={})'.format(ct.stride, padding)
    else:
        conv_ten = convolution(input_ten, filt_ten, padding, strides, dilation) 
        cmd_string = 'tf.nn.convolution(input=input, filter=filter, ' \
                'padding={}, strides={}, dilation_rate={})'.format(
                padding, str(strides), str(dilation))

    conv = tf.squeeze(conv_ten).numpy()
    return conv, cmd_string

