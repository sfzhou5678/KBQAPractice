import tensorflow as tf


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
    self.is_forward = tf.placeholder(tf.bool)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
      # Look up embeddings for inputs.
      self.entities_embeddings = tf.get_variable('entities_embeddings', [entities_vocab_size, embedding_size],
                                                 dtype=tf.float32)

      self.relations_embeddings = tf.get_variable('relations_embeddings', [relations_vocab_size, embedding_size],
                                                  dtype=tf.float32)

      embed_heads = tf.nn.embedding_lookup(self.entities_embeddings, self.heads)
      embed_relations = tf.nn.embedding_lookup(self.relations_embeddings, self.relations)

      nce_weights = tf.get_variable('nce_weights', [entities_vocab_size, embedding_size], dtype=tf.float32, )
      nce_biases = tf.get_variable('nce_biases', [entities_vocab_size], dtype=tf.float32, )

    # 如果有一组数据是h+r=t,那么forward时h+r=t,backward时t-r=h
    embed = tf.cond(self.is_forward,
                    lambda: tf.add(embed_heads, embed_relations),
                    lambda: tf.subtract(embed_heads, embed_relations))
    # embed = tf.add(embed_heads, embed_relations)

    # 由于nce内部做的只是一个LR二分类，所以还可以尝试加一个softmax变成预测问题，配上对应的loss和opt
    # 注: 训练softmax会导致模型训练时间大幅延长，慎用
    softmax_weights = tf.get_variable('softmax_weights', [embedding_size, entities_vocab_size], dtype=tf.float32)
    softmax_biases = tf.get_variable('softmax_biases', [entities_vocab_size], dtype=tf.float32)
    softmax_logits = tf.matmul(embed, softmax_weights) + softmax_biases
    self.softmax_pred = tf.argmax(tf.nn.softmax(softmax_logits), axis=-1)

    softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(self.tails, [batch_size]), logits=softmax_logits, name='softmax_loss'
    )
    self.softmax_loss = tf.reduce_mean(softmax_loss)

    self.loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=self.tails,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=entities_vocab_size))

    correct_prediction = tf.equal(tf.cast(tf.argmax(softmax_logits, -1), dtype=tf.int32),
                                  tf.reshape(self.tails, [batch_size]))
    self.softmax_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    top20_correction_predcition = tf.nn.in_top_k(predictions=softmax_logits,
                                                 targets=tf.reshape(self.tails, [batch_size]), k=20)
    self.softmax_top20_accuracy = tf.reduce_mean(tf.cast(top20_correction_predcition, tf.float32))

    top100_correction_predcition = tf.nn.in_top_k(predictions=softmax_logits,
                                                  targets=tf.reshape(self.tails, [batch_size]), k=100)
    self.softmax_top100_accuracy = tf.reduce_mean(tf.cast(top100_correction_predcition, tf.float32))

    if is_training:
      global_step = tf.contrib.framework.get_or_create_global_step()
      learning_rate = tf.train.exponential_decay(
        config.base_learning_rate,
        global_step,
        300,
        0.98
      )
      self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
      self.softmax_train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.softmax_loss)
