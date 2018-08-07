# Testing memory usage of 1x1 convolution vs matmul
import tensorflow as tf

I = 32 # num input channels
O = 512 # num output channels
B = 10 # batch size
T = 1000 # time steps

filt = tf.get_variable(shape=[I, O], name='filt')
input = tf.random_normal(shape=[B, T, I], name='input')

with tf.variable_scope('conv'):
    conv = tf.nn.convolution(input, tf.expand_dims(filt, 0), 'VALID', [1], [1], 'conv')
with tf.variable_scope('mat'):
    # filt_exp = tf.broadcast_to(f, [7, 32, 512]) # can't use - no gradient op implemented
    filt_exp = tf.stack([filt] * B)
    mm = tf.matmul(input, filt_exp)

def print_tensor_sizes(sess):
    '''
    for each tensor in the graph, print:
    name, shape, size, data_type_size
    '''
    for op in sess.graph.get_operations():
        for ten in op.outputs:
            try:
                fields = (ten.name, str(ten.shape.as_list()),
                ten.shape.num_elements(), ten.dtype.size)
            except ValueError:
                fields = (ten.name, '?', '?', '?')

            print('\t'.join(map(str, fields)))

sess = tf.Session()
print_tensor_sizes(sess)

