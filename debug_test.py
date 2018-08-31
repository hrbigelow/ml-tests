# Find the c++ source code that's executed when we call tf.conv2d

import tensorflow as tf
tf.enable_eager_execution()
import os
os.getpid() 

# now, do sudo gdb -p <pid>
# gdb $ b TFE_Py_FastPathExecute_C

# This is where the op is actually executed
# gdb $ b tensorflow/python/eager/pywrap_tfe_src.cc:2418

# In tensorflow/python/eager/pywrap_tfe_src.cc:2173

# This function corresponds somehow to the function called
# _pywrap_tensorflow.TFE_Py_FastPathExecute, in
# bazel-genfiles/tensorflow/python/ops/gen_nn_ops.py:971

# execute.cc:695 kernel->Run call
# kernel_and_device.cc:95-110 seems to be where the action happens

# gpu_device.cc:492

# At pywrap_tfe_src.cc:2200, we have 

import numpy as np
 
def prod(x):
    p = 1
    for e in x:
        p *= e
    return p

filter_sz = [5, 1, 1]
input_sz = [1, 45, 1]
ndim = len(filter_sz)

filt = np.array(list(range(1, prod(filter_sz) + 1))).reshape(filter_sz)
filt_ten = tf.constant(filt, dtype=tf.float64)
input = np.array(list(range(1, prod(input_sz) + 1))).reshape(input_sz)
input_ten = tf.constant(input, dtype=tf.float64)


conv = tf.nn.conv1d(input_ten, filt_ten, 2, 'SAME')
