import tensorflow as tf

# Code snippets to illustrate how tf.get_variable and tf.variable_scope work.
# tf.get_variable works with a key/value store. keys are names, and the values
# are tf.Variable's. 

# The call to tf.get_variable works in two phases.
# Phase 1. Key Construction
# In order to construct the key, tf.get_variable consults the entire nested scope
# created by tf.variable_scope context managers.  For example, if you have:

with tf.variable_scope('foo'):
    with tf.variable_scope('bar'):
        with tf.variable_scope('bat');
            a = tf.get_variable('car', [1,2,3])

# the key that tf.get_variable will use is 'foo/bar/bat/car'

# Phase 2: Reuse policy
# Each call to tf.variable_scope has a 'reuse' parameter, which can be 'True', 'False',
# or tf.AUTO_REUSE.  This parameter affects how tf.get_variable interacts with the store.

# reuse=True
# tf.get_variable uses the key constructed in phase 1 to look in the store.
# If the store contains an entry, that entry is returned.  Otherwise, tf.get_variable
# raises ValueError: Variable foo/f does not exist, or was not created with tf.get_variable()

# reuse=False
# tf.get_variable expects the key NOT to be in the store.  If it is not, a new variable is
# created, and added to the store under the key.  If it is, raises:
# ValueError: Variable foo/d already exists, disallowed. \
# Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?

# reuse=tf.AUTO_REUSE
# tf.get_variable gracefully handles both situations (key present or absent).  If key is present
# in the store, returns the retrieved variable.  If absent, creates a variable and stores
# it under the key, and returns it.




# Definitions:
# 1: Name-collision: a new variable has same name as existing variable
# 2: Missing-from-storage: a variable is requested from storage but it is not there

# In graph mode, TensorFlow raises an error if:
# 1. reuse=None and Name-collision happens
# 2. reuse=True and Missing-from-storage happens

# Simple test to demonstrate that the reuse parameter indeed
# is used while in Graph mode
with tf.variable_scope("foo", reuse=True) as vs:
    print(vs.reuse) # True

with tf.variable_scope("foo", reuse=False) as vs:
    print(vs.reuse) # False 

with tf.variable_scope("foo", reuse=None) as vs:
    print(vs.reuse) # False

with tf.variable_scope("foo", reuse=tf.AUTO_REUSE) as vs:
    print(vs.reuse) # _ReuseMode.AUTO_REUSE

# Example: when reuse=tf.AUTO_REUSE, TensorFlow retrieves a
# variable from storage if it exists, or creates (and registers it
# in storage) if it does not.  In this case, 'a' is created and stored,
# then 'b' is retrieved from storage, so they are the same
with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    a = tf.get_variable('d', [1,2,3])
    b = tf.get_variable('d', [1,2,3])

assert a == b

# Example. when reuse=None (or False)
with tf.variable_scope("foo", reuse=False):
    # Assume 'foo/d' already exists in the variable store
    a = tf.get_variable('d', [1,2,3])
# Raises:
# ValueError: Variable foo/d already exists, disallowed. \
# Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?

# Example when reuse=True
with tf.variable_scope("foo", reuse=True):
    # Assume 'foo/f' does not exist in the variable store 
    a = tf.get_variable('f', [1,2,3])
# Raises
# ValueError: Variable foo/f does not exist, or was not created with tf.get_variable(). \
# Did you mean to set reuse=tf.AUTO_REUSE in VarScope?


# In Eager mode, the reuse parameter to tf.variable_scope is over-ridden
# to be tf.AUTO_REUSE.  This actually is misleading, though, because in fact
# the behavior of tf.get_variable in Eager mode doesn't use a variable store
# at all.  It does no name checks, nor does it access any storage.




