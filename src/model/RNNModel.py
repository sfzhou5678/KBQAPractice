import tensorflow as tf


class RNNModel(object):


  def __init__(self, config, is_training,is_testing=False):
    max_question_length = 20
    max_candidate_item_size = 16
    max_item_label_length = 24

    max_candidate_relation_size = 128
    max_relation_label_length = 24

    question = tf.placeholder(tf.int32, [None, config.max_question_length], 'question')
    candidate_items = tf.placeholder(tf.int32, [None, config.max_candidate_item_size, config.max_item_label_length],
                                     name='candidate_items')
    candidate_relations = tf.placeholder(tf.int32, [None, config.max_candidate_relation_size, config.max_relation_label_length],
                                         name='candidate_relations')
    if not is_testing:
      # 只有非测试的时候才会提供gt数据
      gt_item=tf.placeholder(tf.int32,[None,1,config.max_item_label_length],name='gt_item')
      gt_relation=tf.placeholder(tf.int32,[None,1,config.max_candidate_relation_size],name='gt_relation')

    # TODO: 1. 将item和relation转化成embedding
    # TODO: 1.1 首先转化成one-hot形式
    # TODO: 1.2 然后在通过CNN，将one-hot编码

    # TODO: 2. 用带ATT的RNN扫描question，第一步提取Item(要有一个输出argmax的函数)，第二部提取R


    # TODO: 3. loss&trainOpt&acc

