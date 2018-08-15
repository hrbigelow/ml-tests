import tensorflow as tf
from tensorflow.nn import convolution
from tensorflow.contrib.nn import conv1d_transpose

def tf_conv(conv_type, input):
    '''implement the equivalent TensorFlow convolution or deconvolution based
    on the parameters in conv_type'''
    def add_dims(ten, *dims):
        for d in dims:
            ten = tf.expand_dims(ten, d)
        return ten

    ct = conv_type
    assert tf.executing_eagerly()
    
    input_ten = add_dims(tf.constant(input), 0, 2)
    filt_ten = add_dims(tf.constant(ct.filter(do_dilate=False)), 1, 2) 
    strides = [ct.stride]
    dilation = [ct.dilation]
    padding = ct.padding_type()
    if padding not in ('VALID', 'SAME'):
        print('tf_conv: cannot execute with {} padding'.format(padding), stderr)
        return np.array([])



    if ct.is_inverse:
        if ct.dilation != 0:
            print('tf_conv: conv1d_transpose doesn\'t support dilation > 0')
            return np.array([])
        conv_ten = conv1d_transpose(input_ten, filt_ten, ct.stride, padding)
    else:
        conv_ten = convolution(input_ten, filt_ten, padding, strides, dilation) 


    conv = tf.squeeze(conv_ten).numpy()
    return conv

