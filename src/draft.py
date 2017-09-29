import json
import numpy as np
import tensorflow as tf
from collections import Counter

from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

# 开始验证nce的负采样sampler

batch_size = 32
num_classes = 100
# labels = np.random.randint(0, num_classes, size=[batch_size, 1])
input_ids = np.random.randint(0, num_classes, size=[batch_size])
# entity_ids = np.random.randint(0, num_classes, size=[batch_size, 1])
entity_ids = np.ones([batch_size, 1], dtype=np.int32)
relation_ids = np.random.randint(0, num_classes, size=[batch_size, 1])

entity_embedding = tf.get_variable('entity_embedding', [num_classes, 32], dtype=tf.float32)
inputs = tf.nn.embedding_lookup(entity_embedding, input_ids)

neg_entity_sampler = candidate_sampling_ops.log_uniform_candidate_sampler(
  true_classes=entity_ids,
  num_true=1,
  num_sampled=64,
  unique=True,
  range_max=num_classes)
sampled, true_expected_count, sampled_expected_count = (
  array_ops.stop_gradient(s) for s in neg_entity_sampler)
sampled = math_ops.cast(sampled, tf.int32)

labels_flat = array_ops.reshape(entity_ids, [-1])
all_ids = array_ops.concat([labels_flat, sampled], 0)
weights = tf.get_variable('weights', [num_classes, 32], dtype=tf.float32)
all_w = embedding_ops.embedding_lookup(
  weights, all_ids, partition_strategy='mod')

true_w = array_ops.slice(
  all_w, [0, 0], array_ops.stack([array_ops.shape(labels_flat)[0], -1]))


def _sum_rows(x):
  """Returns a vector summing up each row of the matrix x."""
  # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
  # a matrix.  The gradient of _sum_rows(x) is more efficient than
  # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
  # we use _sum_rows(x) in the nce_loss() computation since the loss
  # is mostly used for training.
  cols = array_ops.shape(x)[1]
  ones_shape = array_ops.stack([cols, 1])
  ones = array_ops.ones(ones_shape, x.dtype)
  return array_ops.reshape(math_ops.matmul(x, ones), [-1])


dim = array_ops.shape(true_w)[1:2]
new_true_w_shape = array_ops.concat([[-1, 1], dim], 0)
row_wise_dots = math_ops.multiply(
  array_ops.expand_dims(inputs, 1),
  array_ops.reshape(true_w, new_true_w_shape))
# We want the row-wise dot plus biases which yields a
# [batch_size, num_true] tensor of true_logits.
dots_as_matrix = array_ops.reshape(row_wise_dots,
                                   array_ops.concat([[-1], dim], 0))
true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, 1])

sampled_w = array_ops.slice(
  all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
sampled_logits = math_ops.matmul(
  inputs, sampled_w, transpose_b=True)

subtract_log_q = True
if subtract_log_q:
  # Subtract log of Q(l), prior probability that l appears in sampled.
  true_logits -= math_ops.log(true_expected_count)
  sampled_logits -= math_ops.log(sampled_expected_count)

out_logits = array_ops.concat([true_logits, sampled_logits], 1)
# true_logits is a float tensor, ones_like(true_logits) is a float tensor
# of ones. We then divide by num_true to ensure the per-example labels sum
# to 1.0, i.e. form a proper probability distribution.
out_labels = array_ops.concat([
  array_ops.ones_like(true_logits) / 1,
  array_ops.zeros_like(sampled_logits)
], 1)

sampled_losses = tf.nn.softmax_cross_entropy_with_logits(labels=out_labels,
                                                          logits=out_logits)

# neg_relation_sampler = candidate_sampling_ops.log_uniform_candidate_sampler(
#   true_classes=relation_ids,
#   num_true=1,
#   num_sampled=64,
#   unique=True,
#   range_max=num_classes)



sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
  tf.global_variables_initializer().run()
  for i in range(2):
    all_id = sess.run(all_ids)
    all_ws = sess.run(all_w)
    true_ws = sess.run(true_w)
    sampled_ws = sess.run(sampled_w)
    samples_logits_ = sess.run(sampled_logits)
    true_logits_ = sess.run(true_logits)
    output_logits_,output_labels_=sess.run([out_logits,out_labels])
    loss=sess.run(sampled_losses)

    print(all_id)
    print(all_ws.shape)
    print(true_ws.shape)
    print(sampled_ws.shape)
    print(samples_logits_.shape)
    print(true_logits_.shape)
    print(output_logits_.shape,output_labels_.shape)
    print(loss)
    # neg_relation_samples = sess.run(neg_relation_sampler)
