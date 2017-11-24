import tensorflow as tf

cnn_w1 = tf.get_variable("cnn_w1", [16, 32, 32, 16])
shapes=cnn_w1.get_shape().as_list()
fc_w1 = tf.get_variable("fc_w1", shapes)

res = tf.nn.tanh(tf.multiply(cnn_w1, fc_w1))


with tf.Session() as sess:
  tf.global_variables_initializer().run()
  print(sess.run(res))
