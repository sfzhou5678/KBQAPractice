import os
import tensorflow as tf
from src.transe.transe_data_helper import *
from src.transe.TransEModel import TransEModel
from src.configs import *
from src.tools.reader import get_vocabulary, get_train_test_ids

if __name__ == '__main__':
  config = TransEConfig()
  data_folder = r'F:\FBData'

  # ========数据路径设置==========
  file_path = os.path.join(data_folder, 'SmallBase_gama')
  selected_twice_data_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.twice.se')

  selected_ids_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.id.pkl')
  selected_train_ids_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.train.id.pkl')
  selected_test_ids_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.test.id.pkl')

  useful_entity_counter_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.wordCounter.useful.pkl')
  relation_counter_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.relationCounter.pkl')

  entity_vocab_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.entityVocab.pkl')
  relation_vocab_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.relationVocab.pkl')

  # ========提取关键信息==========

  relation_vocab = get_vocabulary(relation_vocab_path, relation_counter_path, percent=1.0)
  entity_vocab = get_vocabulary(entity_vocab_path, useful_entity_counter_path, percent=0.95)

  config.relations_vocab_size = len(relation_vocab)
  config.entities_vocab_size = len(entity_vocab)

  train_ids, test_ids = get_train_test_ids(selected_twice_data_path, selected_ids_path,
                                           selected_train_ids_path, selected_test_ids_path,
                                           entity_vocab, relation_vocab)
  print('ids准备完毕')

  # ==========转化成tensor========
  train_head_batch, train_r_batch, train_t_batch = batch_producer(train_ids, config.batch_size)
  test_head_batch, test_r_batch, test_t_batch = batch_producer(test_ids, config.batch_size)
  print('data batch准备完毕')

  initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
  with tf.name_scope('Train'):
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
      model = TransEModel(config=config, is_training=True)

  with tf.name_scope('Test'):
    # 注意这里的variable_scope要和训练集一致，而且train的reuse=None而valid和test的reuse都是True
    with tf.variable_scope("Model", reuse=True):
      test_model = TransEModel(config=config, is_training=False)
  print('模型准备完毕')

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(200000):
      head, relations, tail = sess.run([train_head_batch, train_r_batch, train_t_batch])
      # _, loss = sess.run([model.train_op, model.loss],
      #                    {model.heads: head, model.relations: relations, model.tails: tail})
      _, softmax_loss, _, loss, = sess.run([model.softmax_train_op, model.softmax_loss,
                                            model.train_op, model.loss,
                                            ],
                                           {model.heads: head, model.relations: relations,
                                            model.tails: tail})
      if step % 400 == 0:
        softmax_pred, softmax_acc, softmax_top20_acc, softmax_top100_acc = sess.run(
          [model.softmax_pred, model.softmax_accuracy,
           model.softmax_top20_accuracy, model.softmax_top100_accuracy, ],
          {model.heads: head, model.relations: relations, model.tails: tail})

        test_head, test_relation, test_tail = sess.run([test_head_batch, test_r_batch, test_t_batch])
        test_loss, test_softmax_loss = sess.run([test_model.loss, test_model.softmax_loss, ],
                                                {test_model.heads: test_head, test_model.relations: test_relation,
                                                 test_model.tails: test_tail})
        test_softmax_pred, test_softmax_acc, test_softmax_top20_acc, test_softmax_top100_acc = sess.run(
          [test_model.softmax_pred, test_model.softmax_accuracy,
           test_model.softmax_top20_accuracy, test_model.softmax_top100_accuracy, ],
          {test_model.heads: test_head, test_model.relations: test_relation, test_model.tails: test_tail})

        print('==============[%d] %.4f %.4f\t %.4f %.4f==============' % (step, loss, softmax_loss,
                                                                          test_loss, test_softmax_loss))
        print('[softmax acc: %.3f]\t[top20 acc: %.3f]\t[top100 acc: %.3f]' % (
          softmax_acc, softmax_top20_acc, softmax_top100_acc))

        print('[t-softmax acc: %.3f]\t[t-top20 acc: %.3f]\t[t-top100 acc: %.3f]' % (
          test_softmax_acc, test_softmax_top20_acc, test_softmax_top100_acc))

    coord.request_stop()
    coord.join(threads)
