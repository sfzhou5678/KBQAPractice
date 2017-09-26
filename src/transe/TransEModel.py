import tensorflow as tf
import math


class TransEModel:
  def __init__(self, config, is_training=False):
    batch_size = config.batch_size
    entities_vocab_size = config.entities_vocab_size
    relations_vocab_size = config.relations_vocab_size
    embedding_size = config.embedding_size

    num_sampled = config.num_sampled

    # Input data.
    self.heads = tf.placeholder(tf.int32, shape=[batch_size])
    self.relations = tf.placeholder(tf.int32, shape=[batch_size])
    # if is_training:
    self.tails = tf.placeholder(tf.int32, shape=[batch_size, 1])

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
      # Look up embeddings for inputs.
      entities_embeddings = tf.Variable(
        tf.random_uniform([entities_vocab_size, embedding_size], -1.0, 1.0))
      relations_embeddings = tf.Variable(
        tf.random_uniform([relations_vocab_size, embedding_size], -1.0, 1.0))

      embed_heads = tf.nn.embedding_lookup(entities_embeddings, self.heads)
      # embed_tails = tf.nn.embedding_lookup(entities_embeddings, tails)  # target不需要做embedding

      embed_relations = tf.nn.embedding_lookup(relations_embeddings, self.relations)
      # Construct the variables for the NCE loss
      # TODO nce weights是什么？
      nce_weights = tf.Variable(
        tf.truncated_normal([entities_vocab_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
      nce_biases = tf.Variable(tf.zeros([entities_vocab_size]))

    embed = tf.add(embed_heads, embed_relations)
    if is_training:
      # todo NCELoss详解
      self.loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=self.tails,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=entities_vocab_size))
    else:
      weights=tf.transpose(nce_weights,[1,0])
      logits = tf.matmul(embed, weights) + nce_biases
      final_prob = tf.nn.softmax(logits)

      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.reshape(self.tails,[batch_size]), logits=logits, name="sampled_losses")
      self.loss = tf.reduce_mean(loss)

    if is_training:
      # Construct the SGD optimizer using a learning rate of 1.0.
      self.train_op = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)
