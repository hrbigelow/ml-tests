import tensorflow as tf
import convtype as ctyp
import numpy as np
from tensorflow.nn import convolution
from tensorflow.contrib.nn import conv1d_transpose
from sys import stderr


def conv(input, matrix_sz, filt, inv, st, pad, dil):
    '''implement the equivalent TensorFlow convolution or deconvolution based
    on the parameters in conv_type'''
    def add_dims(ten, *dims):
        for d in dims:
            ten = tf.expand_dims(ten, d)
        return ten

    assert tf.executing_eagerly()
    
    # B x W x C
    input_ten = add_dims(tf.constant(input, dtype=tf.float64), 0, 2)
    filt_ten = add_dims(tf.constant(filt, dtype=tf.float64), 1, 2) 

    strides = [st]
    dilation = [dil]

    if pad not in ('VALID', 'SAME'):
        return (np.array([]), 'Not executed.  Padding must be VALID or SAME')

    if not inv and dil > 1 and st > 1:
        return (np.array([]), 'Not executed. stride > 1 and dilation > 1')

    with tf.device('/cpu:0'):
        try:
            if inv:
                output_shape = tf.constant([1, matrix_sz, 1])
                conv_ten = conv1d_transpose(input_ten, filt_ten, output_shape, st, pad)
                cmd = 'tf.contrib.nn.conv1d_transpose(value=input, filter=filter, ' \
                        'output_shape={}, stride={}, padding={})'.format(
                                str(output_shape.numpy()), st, pad)
            else:
                conv_ten = convolution(input_ten, filt_ten, pad, strides, dilation) 
                cmd = 'tf.nn.convolution(input=input, filter=filter, ' \
                        'padding={}, strides={}, dilation_rate={})'.format(
                        pad, str(strides), str(dilation))
        except:
            return np.array([]), 'Not executed'


        # only squeeze B and C, leaving W intact.
        conv = tf.squeeze(conv_ten, [0,2]).numpy()
    return conv, cmd
