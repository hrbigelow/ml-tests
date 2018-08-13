# Testing memory usage of 1x1 convolution vs matmul
import tensorflow as tf

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

I = 32 # num input channels
O = 512 # num output channels
B = 10 # batch size
T = 1000 # time steps

with tf.variable_scope('zzzinput'):
    filt = tf.get_variable(shape=[I, O], name='filt')
    input = tf.random_normal(shape=[B, T, I], name='input')

with tf.name_scope('cnv1x1'):
    conv = tf.nn.convolution(input, tf.expand_dims(filt, 0), 'VALID', [1], [1], 'conv')
with tf.name_scope('matmul'):
    # filt_exp = tf.broadcast_to(f, [7, 32, 512]) # can't use - no gradient op implemented
    filt_exp = tf.stack([filt] * B)
    mm = tf.matmul(input, filt_exp)

# Both approaches give the same result
run_meta = tf.RunMetadata()
run_opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
        output_partition_graphs=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#all_equal = tf.reduce_all(tf.equal(conv, mm))
#print('conv equal to mm? ' + str(sess.run(all_equal,
#    options=run_opts, run_metadata=run_meta)))

print(sess.run(tf.reduce_max(conv), options=run_opts, run_metadata=run_meta))
#print_tensor_sizes(sess)
print(str(run_meta))

