import tensorflow as tf
import numpy as np
from src.tools.common_tools import cos_similarity, build_decoder_cell_with_att, get_topk_acc
from src.model.cnn_utils import *
from src.configs import RNNModelConfig


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
      # 要求gt_items和gt_relations都是正确答案在candidate中的位置, 所以shape为[batch_size]
      self.gt_items = tf.placeholder(tf.int32, [None], name='gt_items')
      self.gt_relations = tf.placeholder(tf.int32, [None], name='gt_relations')

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
    candidate_items_onehot = tf.one_hot(tf.reshape(self.candidate_items,
                                                   [config.batch_size * config.max_candidate_item_size,
                                                    config.max_item_label_length]), config.char_vocab_size)
    candidate_items_onehot = tf.expand_dims(candidate_items_onehot, -1)

    # 1.2 然后在通过CNN，将one-hot编码转换成和relation通embeddingSize的向量
    # TODO: 现在CharCNN是用2层卷积做的，下次尝试按论文中的FC做法
    candidate_item_embeddings = self._embed_item_entity(candidate_items_onehot,
                                                        config.char_vocab_size, config.entity_embedding_size,
                                                        "char_level_cnn_encoder", is_training)
    # candidate_item_embeddings的期望shape=[config.batch_size, config.max_candidate_item_size,config.entity_embedding_size]
    candidate_item_embeddings = tf.reshape(candidate_item_embeddings,
                                           [config.batch_size, config.max_candidate_item_size,
                                            config.entity_embedding_size])

    # 2 用带Att的LSTM扫描问题
    encoder_outputs, encoder_final_state = self._encode_question(question_embedding, config, is_testing)

    # 3 带Att的RNN Decoder，提取预测处的item，relation以及计算出的所有候选数据是正确答案的概率
    # 3.1 第一步提取Item(要有一个输出argmax的函数)
    self.pred_items, self.pred_relations, \
    self.item_similarities, self.relation_similarities = self._pred(candidate_item_embeddings,
                                                                    candidate_relation_embeddings,
                                                                    encoder_outputs, encoder_final_state, config,
                                                                    is_training)

    # self.loss, self.train_opt = self._build_loss(self.item_similarities, self.relation_similarities,
    #                                                       self.gt_items, self.gt_relations)

    # TODO: 3. loss&trainOpt&acc
    self.item_loss, self.relation_loss = self._build_loss(
      self.item_similarities, self.relation_similarities,
      self.gt_items, self.gt_relations, config)

    self.item_train_opt, self.relation_train_opt = self._build_train_opt(self.item_loss, self.relation_loss, config)
    self.item_acc = get_topk_acc(self.item_similarities, self.gt_items, topk=1)
    self.relation_acc = get_topk_acc(self.relation_similarities, self.gt_relations, topk=1)

  def _pred(self, candidate_item_embeddings, candidate_relation_embeddings,
            encoder_outputs, encoder_final_state, config, is_training):
    """
    根据encoder的outputs和finalState从候选items中选择最有可能的item
    :param encoder_outputs:
    :param encoder_final_state:
    :param config:
    :param is_training:
    :return:
    """
    decoder_cell, decoder_init_state = build_decoder_cell_with_att(encoder_outputs, encoder_final_state,
                                                                   config.batch_size, config.max_question_length,
                                                                   config.rnn_layers, config.hidden_size,
                                                                   config.keep_prob,
                                                                   is_training)
    # fixme:不知道应该和output还是和state比较相似度，目前先以output来处理(因为这个output本质上应该是hState)
    GO_ID_embedding = tf.ones([config.batch_size, config.entity_embedding_size])
    state = decoder_init_state
    with tf.variable_scope("RNN"):
      for time_step in range(2):
        if time_step > 0:
          tf.get_variable_scope().reuse_variables()
        if time_step == 0:
          (cell_output, state) = decoder_cell(GO_ID_embedding, state)
          h1 = cell_output

          ## item_similarities.shape=[batch_size,max_item_size]
          item_similarities = self._calc_similarity(h1, candidate_item_embeddings, config.batch_size,
                                                    config.max_candidate_item_size,
                                                    mode='cos')
          pred_items = tf.argmax(item_similarities, axis=-1)  # pred_items.shape=[batchsize]

        if time_step == 1:
          if is_training:
            pred_item_embedding = self._lookup_embedding(candidate_item_embeddings, self.gt_items, config.batch_size)
          else:
            pred_item_embedding = self._lookup_embedding(candidate_item_embeddings, pred_items, config.batch_size)
          (cell_output, state) = decoder_cell(pred_item_embedding, state)
          h2 = cell_output
          relation_similarities = self._calc_similarity(h2, candidate_relation_embeddings, config.batch_size,
                                                        config.max_candidate_relation_size, mode='cos')
          pred_relations = tf.argmax(relation_similarities, axis=-1)  ## pred_items.shape=[batchsize]

    return pred_items, pred_relations, item_similarities, relation_similarities

  def _lookup_embedding(self, candidate_item_embeddings, pred_items, batch_size):
    """
    现在用的是基于for的embedding获取方式，如果有条件要换成基于矩阵的
    :param candidate_item_embeddings:
    :param pred_items:
    :param batch_size:
    :return:
    """
    pred_item_embedding = []
    for i in range(batch_size):
      index = pred_items[i]
      pred_item_embedding.append(candidate_item_embeddings[i, index])
    pred_item_embedding = tf.convert_to_tensor(pred_item_embedding)

    return pred_item_embedding

  def _calc_similarity(self, h1, candidate_item_embeddings, batch_size, max_candidate_size, mode='cos'):
    """
    计算某个h与candidateEmbedding之间的相似度
    默认为余弦相似度
    :param h1:
    :param candidate_item_embeddings:
    :param mode:
    :return:
    """
    if mode == 'cos':
      similarities = []
      for i in range(max_candidate_size):
        similarity = cos_similarity(h1, candidate_item_embeddings[:, i, :])
        similarities.append(similarity)
      similarities = tf.convert_to_tensor(similarities)
      # similarities = tf.reshape(similarities, [batch_size, max_candidate_size])
      similarities = tf.transpose(similarities)
      return similarities
    else:
      raise Exception('Mode Error')

  def _encode_question(self, question_embedding, config, is_training):
    batch_size = config.batch_size
    bi_lstm = config.bi_lstm
    rnn_layers = config.rnn_layers
    hidden_size = config.hidden_size
    keep_prob = config.keep_prob
    max_question_length = config.max_question_length

    with tf.variable_scope('encoder') as encoder_scope:
      def build_cell(hidden_size):
        def get_single_cell(hidden_size, keep_prob):
          cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
          if keep_prob < 1:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
          return cell

        cell = tf.nn.rnn_cell.MultiRNNCell(
          [get_single_cell(hidden_size, keep_prob) for _ in range(rnn_layers)])

        return cell

      if not bi_lstm:
        encoder_cell = build_cell(hidden_size)
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
          encoder_cell, question_embedding,
          sequence_length=tf.constant(max_question_length, shape=[batch_size], dtype=tf.int32),
          dtype=tf.float32, scope=encoder_scope)
        return encoder_outputs, encoder_final_state
      else:
        encoder_cell = build_cell(hidden_size / 2)
        bw_encoder_cell = build_cell(hidden_size / 2)
        encoder_outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
          encoder_cell, bw_encoder_cell,
          question_embedding,
          sequence_length=tf.constant(max_question_length, shape=[batch_size], dtype=tf.int32),
          dtype=tf.float32, scope=encoder_scope)

        state = []
        for i in range(rnn_layers):
          fs = fw_state[i]
          bs = bw_state[i]
          encoder_final_state_c = tf.concat((fs.c, bs.c), 1)
          encoder_final_state_h = tf.concat((fs.h, bs.h), 1)
          encoder_final_state = tf.nn.rnn_cell.LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h)
          state.append(encoder_final_state)
        encoder_final_state = tuple(state)

        encoder_outputs = tf.maximum(encoder_outputs[0], encoder_outputs[1])
        return encoder_outputs, encoder_final_state

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
      # fixme: 论文中套上了FC？
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

  def _build_loss(self, item_similarities, relation_similarities, gt_items, gt_relations, config):
    item_softmax = tf.nn.softmax(item_similarities)
    relation_softmax = tf.nn.softmax(relation_similarities)

    gt_items_onehot = tf.one_hot(gt_items, config.max_candidate_item_size)
    gt_relations_onehot = tf.one_hot(gt_relations, config.max_candidate_relation_size)

    item_cross_entropy = - tf.reduce_sum(gt_items_onehot * tf.log(item_softmax))/config.batch_size
    relation_cross_entropy = - tf.reduce_sum(gt_relations_onehot * tf.log(relation_softmax))/config.batch_size

    return item_cross_entropy, relation_cross_entropy

  def _build_train_opt(self, item_cross_entropy, relation_cross_entropy, config):
    item_train_opt = tf.train.AdamOptimizer(config.base_learning_rate).minimize(item_cross_entropy)
    relation_train_opt = tf.train.AdamOptimizer(config.base_learning_rate).minimize(relation_cross_entropy)

    return item_train_opt, relation_train_opt


def main():
  pass


if __name__ == '__main__':
  main()
