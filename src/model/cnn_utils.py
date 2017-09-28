import tensorflow as tf

slim = tf.contrib.slim


def batch_norm(inputs, is_training, decay=0.999, epsilon=0.001):
  # fixme: 在val或test时将is_training置为False则会导致acc基本始终为0，若为True则可以得到期望效果
  # output = slim.batch_norm(inputs, decay=decay, epsilon=epsilon, is_training=is_training)
  output = slim.batch_norm(inputs, decay=decay, epsilon=epsilon)

  return output


def weight_variable(shape):
  return tf.get_variable('weight', shape,
                         initializer=tf.truncated_normal_initializer(stddev=0.1))


def bias_variable(shape):
  return tf.get_variable('bias', shape, initializer=tf.constant_initializer(0.01))


def conv_2d(input, w_shape, b_shape, strides, name, is_training, norm, padding='SAME',
            act_func=tf.nn.relu):
  with tf.variable_scope(name):
    w = weight_variable(w_shape)
    b = bias_variable(b_shape)
    conv = tf.nn.conv2d(input, w, strides=strides, padding=padding)

    if norm:
      conv = batch_norm(conv, is_training=is_training)
    else:
      conv = tf.nn.bias_add(conv, b)

    if act_func != None:
      conv = act_func(conv)

    return conv


def max_pool_2d(x, ksize, strides, name, padding='SAME'):
  with tf.variable_scope(name):
    return tf.nn.max_pool(x, ksize=ksize,
                          strides=strides, padding=padding)


def fully_connected(name, input, w_shape, b_shape, global_step, norm, need_dropout=False, keep_prob=1.0,
                    act_function=tf.nn.relu):
  with tf.variable_scope(name):
    w = weight_variable(w_shape)
    b = bias_variable(b_shape)
    fc = tf.matmul(input, w)
    if norm:
      fc, _ = batch_norm(fc, tf.constant(False, dtype=tf.bool), b, global_step)
    else:
      fc += b

    if act_function != None:
      fc = act_function(fc)

    # fc = tf.cond(need_dropout, lambda:  tf.nn.dropout(fc, keep_prob), lambda: fc)
    if need_dropout:
      fc = tf.nn.dropout(fc, keep_prob)
    return fc
