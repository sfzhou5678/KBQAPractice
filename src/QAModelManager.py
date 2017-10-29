import tensorflow as tf
from src.model.CNNModel import CNNModel
import numpy as np


class QAModelManager(object):
  def __init__(self, config):
    # with tf.name_scope('Test'):
    #   with tf.variable_scope("Model", reuse=True):
    #     self.model = CNNModel(config, is_training=False, is_test=True)
    pass

  def calc_similarity(self, question_ids, topic_id,
                      candidate_relation, candidate_ans, is_forward):
    """
    调用self.model返回每个ans的相似度
    :param question_ids:
    :param topic_id:
    :param candidate_relation:
    :param candidate_ans:
    :param is_forward:
    :return:
    """

    # TODO 调用Model做计算，现在先临时返回一个随机值

    return np.random.uniform(-1, 1, [len(candidate_ans)])
