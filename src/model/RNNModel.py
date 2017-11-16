import tensorflow as tf
import numpy as np
from src.model.cnn_utils import *


class RNNModel(object):
  def __init__(self, config, is_training, is_testing=False):
    self.question = tf.placeholder(tf.int32, [None, config.max_question_length], 'question')
    self.candidate_items = tf.placeholder(tf.int32,
                                          [None, config.max_candidate_item_size, config.max_item_label_length],
                                          name='candidate_items')

    # relation由于数量较少，而且R不存在label，所以直接赋予TransE训练的结果
    self.candidate_relations = tf.placeholder(tf.int32, [None, config.max_candidate_relation_size],
                                              name='candidate_relations')
    if not is_testing:
      # 只有非测试的时候才会提供gt数据
      gt_item = tf.placeholder(tf.int32, [None, 1, config.max_item_label_length], name='gt_item')
      gt_relation = tf.placeholder(tf.int32, [None, 1], name='gt_relation')

    with tf.device("/cpu:0"):
      # 总共有文本word，entity以及relation三个词汇表
      # char = [i for i in range(config.char_vocab_size)]
      # self.char_embeddings = tf.one_hot(char, config.char_vocab_size, name='char_embeddings')
      self.relation_embeddings = tf.get_variable('relation_embeddings',
                                                 [config.relations_vocab_size, config.entity_embedding_size],
                                                 dtype=tf.float32,
                                                 # trainable=False
                                                 )
      candidate_relation_embeddings = tf.nn.embedding_lookup(self.relation_embeddings, self.candidate_relations)

    question_embedding = tf.one_hot(self.question, config.char_vocab_size)

    # 将items从char编码成向量：
    # 1.1 首先转化成one-hot形式
    candidate_items_onehot = tf.one_hot(self.candidate_items, config.char_vocab_size)
    candidate_items_onehot = tf.reshape(candidate_items_onehot,
                                        [config.batch_size * config.max_candidate_item_size,
                                         config.max_item_label_length, config.char_vocab_size, 1])
    # 1.2 然后在通过CNN，将one-hot编码转换成和relation通embeddingSize的向量
    # TODO: 现在CharCNN是用2层卷积做的，下次尝试按论文中的FC做法
    candidate_item_embeddings = self._embed_item_entity(candidate_items_onehot,
                                                        config.char_vocab_size, config.entity_embedding_size,
                                                        "char_level_cnn_encoder", is_training)
    # candidate_item_embeddings的期望shape=[config.batch_size, config.max_candidate_item_size,config.entity_embedding_size]
    candidate_item_embeddings = tf.reshape(candidate_item_embeddings,
                                           [config.batch_size, config.max_candidate_item_size,
                                            config.entity_embedding_size])

    # TODO: 2. 用带ATT的RNN扫描question，第一步提取Item(要有一个输出argmax的函数)，第二部提取R

    # TODO: 3. loss&trainOpt&acc

  def _embed_item_entity(self, inputs, char_vocab_size, output_latent_vec_size, name, is_training):
    """
    将onehot形式的items转换成定长向量
    :param candidate_items_onehot: 
    :return: 
    """
    norm = True

    DEPTH1 = 16
    DEPTH2 = DEPTH1 * 2

    with tf.variable_scope(name, reuse=not is_training):
      # fixme: 论文中并不是用maxpooling而是FC？
      network = conv_2d(inputs, [3, char_vocab_size, 1, DEPTH1], [DEPTH1], [1, 1, 1, 1], 'layer1-conv1',
                        norm=norm,
                        is_training=is_training)
      # network = max_pool_2d(network, [1, 1, 1, 1], [1, 1, 1, 1], 'layer1-pool1')

      network = conv_2d(network, [3, 3, DEPTH1, DEPTH2], [DEPTH2], [1, 1, 1, 1], 'layer2-conv2',
                        norm=norm, is_training=is_training)
      # network = max_pool_2d(network, [1, 2, 2, 1], [1, 2, 2, 1], 'layer2-pool1')

      # 最后将CNN产生的值通过全局平均池化，再通过全连接层产生latent vector
      net = slim.avg_pool2d(network, network.get_shape()[1:3], padding='VALID', scope='AvgPool')
      # 这里不能加is_training=false，如果加了就会导致val时所有cos均为1 (原因未知，但是官方IncepResnetV2中也是恒为true的)
      net = slim.dropout(net, 0.5, is_training=is_training, scope='Dropout')
      net = slim.flatten(net)

      latent_vec = slim.fully_connected(net, output_latent_vec_size, activation_fn=None, scope='latent_vec')

      return latent_vec


def main():
  from src.configs import RNNModelConfig
  config = RNNModelConfig()
  model = RNNModel(config=config, is_training=True, is_testing=False)

  with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # char_embeddings=sess.run(model.char_embeddings)
    # print(char_embeddings)
    for i in range(2):
      question = np.reshape(range(config.batch_size * config.max_question_length),
                            [config.batch_size, config.max_question_length])
      candidate_items = np.reshape(
        range(config.batch_size * config.max_candidate_item_size * config.max_item_label_length),
        [config.batch_size, config.max_candidate_item_size, config.max_item_label_length])
      candidate_relations = np.reshape(range(config.batch_size * config.max_candidate_relation_size),
                                       [config.batch_size, config.max_candidate_relation_size])

      # candidate_item_embeddings = sess.run(
      #   model.candidate_item_embeddings,
      #   {model.question: question,
      #    model.candidate_items: candidate_items,
      #    model.candidate_relations: candidate_relations})


if __name__ == '__main__':
  main()
