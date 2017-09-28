import os
import tensorflow as tf
from src.transe.transe_data_helper import *
from src.transe.TransEModel import TransEModel
from src.transe.transe_config import *

if __name__ == '__main__':
  config = TransEConfig()

  data_folder = r'D:\DeeplearningData\NLP-DATA\英文QA\splitFB'
  training_file = os.path.join(data_folder, 'train.1000.only_entity.txt')
  traininig_ids_file = os.path.join(data_folder, 'train.ids.vocab%d+%d.only_entity.pkl' % (
    config.entities_vocab_size, config.relations_vocab_size))
  test_file = os.path.join(data_folder, 'test.1000.only_entity.txt')
  test_ids_file = os.path.join(data_folder, 'test.ids.vocab%d+%d.only_entity.pkl' % (
    config.entities_vocab_size, config.relations_vocab_size))

  files = [training_file, test_file]

  entities_word2id, entities_id2word, relations_word2id, relations_id2word = build_vocab(files,
                                                                                         config.entities_vocab_size,
                                                                                         config.relations_vocab_size)
  training_ids = build_ids(training_file, traininig_ids_file, entities_word2id, relations_word2id)
  test_ids = build_ids(test_file, test_ids_file, entities_word2id, relations_word2id)

  train_head_batch, train_r_batch, train_t_batch = batch_producer(training_ids, config.batch_size)
  test_head_batch, test_r_batch, test_t_batch = batch_producer(test_ids, config.batch_size)

  initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
  with tf.name_scope('Train'):
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
      model = TransEModel(config=config, is_training=True)

  with tf.name_scope('Test'):
    # 注意这里的variable_scope要和训练集一致，而且train的reuse=None而valid和test的reuse都是True
    with tf.variable_scope("Model", reuse=True):
      test_model = TransEModel(config=config, is_training=False)

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(300000):
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

        # test_head, test_relation, test_tail = sess.run([test_head_batch, test_r_batch, test_t_batch])
        # # test_loss, test_softmax_loss = sess.run([test_model.loss, test_model.softmax_loss, ],
        # #                                         {test_model.heads: test_head, test_model.relations: test_relation,
        # #                                          test_model.tails: test_tail})
        #
        # # 用训练集数据做测试，验证test函数是否有误
        # test_loss, test_softmax_loss = sess.run([test_model.loss, test_model.softmax_loss, ],
        #                                         {test_model.heads: head, test_model.relations: relations,
        #                                          test_model.tails: tail})

        # print('[%d] %.4f, %.4f' % (step, loss, test_loss))

    coord.request_stop()
    coord.join(threads)
