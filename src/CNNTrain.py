import os
import tensorflow as tf
import time
import heapq
from src.configs import CNNModelConfig
from src.model.CNNModel import CNNModel
from src.tools.db_to_model_data import triples_to_tfrecords, get_entity_vocabulary, get_word_vocabulary, \
  get_pretrained_entity_embeddings, get_tf_data


def main():
  # 首先在main中利用DB将原始的问答对转化成Model所需的{question, topicEntity，trueAns，candidateAns}
  wikidata_folder = r'F:\WikiData'
  selected_wikidata_file_path = os.path.join(wikidata_folder, 'selected-latest-all.OnlyRelevant.data')

  transe_data_save_path = os.path.join(wikidata_folder, 'transE.OnlyRelevant.data')
  item_counter_save_path = os.path.join(wikidata_folder, 'item.counter.OnlyRelevant.cnt')
  relation_counter_save_path = os.path.join(wikidata_folder, 'relation.counter.OnlyRelevant.cnt')

  item_vocab_save_path = os.path.join(wikidata_folder, 'item.vocab.OnlyRelevant.voc')
  relation_vocab_save_path = os.path.join(wikidata_folder, 'relation.vocab.OnlyRelevant.voc')

  pretrained_wordvec_save_path = os.path.join(wikidata_folder, 'glove.6B.100d.txt')
  word_vocab_save_path = os.path.join(wikidata_folder, 'word.vocab.glove.100.voc')
  word_embedding_save_path = os.path.join(wikidata_folder, 'word.embedding.glove.100.emd')

  item_embeddings_path = os.path.join(wikidata_folder, 'item_embeddings.npy')
  relation_embeddings_path = os.path.join(wikidata_folder, 'relation.embeddings.emd')

  # 首先在main中利用DB将原始的问答对转化成Model所需的{question, topicEntity，trueAns，candidateAns}
  # db = DBManager(host='192.168.1.139', port=3306, user='root', psd='1405', db='kbqa')
  #
  # train_file_path = '../../data/trains_ansEntity_fixConnectErr.txt'
  # train_triples_candidate_path = r'F:\WikiData\ForFun\train.both.triples+candidate.txt'
  # test_file_path = '../../data/ts_ansEntity_raw.txt'
  # test_triples_candidate_path = r'F:\WikiData\ForFun\test.both.triples+candidate.txt'
  #
  # qid_to_triples(db, train_file_path, train_triples_candidate_path)
  # qid_to_triples(db, test_file_path, test_triples_candidate_path)
  #
  # db.close()

  # 从原30G文件中提取三元组,总共提到了2300W多对
  # prepare_transe_data(selected_wikidata_file_path, transe_data_save_path)

  # 统计三元组中item和relation的词频
  # build_entities_counter(transe_data_save_path, item_counter_save_path, relation_counter_save_path)


  # 构建entities的词汇表，分别列在item和relation两个表中
  # 每个词汇表的0号位都是UNK
  relation_vocab = get_entity_vocabulary(relation_counter_save_path, relation_vocab_save_path, UNK='RELATION_UNK',
                                         percent=1.0)

  item_vocab = get_entity_vocabulary(item_counter_save_path, item_vocab_save_path, UNK='ITEM_UNK', percent=0.80)

  word_vocab, word_embedding = get_word_vocabulary(pretrained_wordvec_save_path,
                                                   word_vocab_save_path,
                                                   word_embedding_save_path,
                                                   UNK='WORD_UNK', PAD='PAD')

  config = CNNModelConfig()
  # 然后配合各vocab，将rawModelData转化成真正的ModelData
  triples_to_tfrecords(r'F:\WikiData\ForFun\train.both.triples+candidate.txt', r'F:\WikiData\ForFun',
                       word_vocab, item_vocab, relation_vocab, config.max_question_length,
                       padding_id=1,
                       num_samples=config.num_sampled,
                       mode='train')

  triples_to_tfrecords(r'F:\WikiData\ForFun\test.both.triples+candidate.txt', r'F:\WikiData\ForFun',
                       word_vocab, item_vocab, relation_vocab, config.max_question_length,
                       padding_id=1,
                       mode='test')

  question, topic_entity, true_ans, true_relation_list, neg_ans, neg_relation = get_tf_data(
    r'F:\WikiData\ForFun\train.forward.%d.tfrecords' % config.num_sampled, config.max_question_length,
    config.num_sampled, mode='train')

  bf_question, bf_topic_entity, bf_true_ans, bf_true_relation_list, bf_neg_ans, bf_neg_relation = get_tf_data(
    r'F:\WikiData\ForFun\train.backward.%d.tfrecords' % config.num_sampled, config.max_question_length,
    config.num_sampled, mode='train')

  test_question, test_topic_entity, \
  test_true_ans_list, test_true_relation_list, test_candidate_ans, test_candidate_realtion = get_tf_data(
    r'F:\WikiData\ForFun\test.forward.%d.tfrecords' % config.num_sampled, config.max_question_length,
    config.num_sampled, mode='test')

  bf_test_question, bf_test_topic_entity, \
  bf_test_true_ans_list, bf_test_true_relation_list, bf_test_candidate_ans, bf_test_candidate_realtion = get_tf_data(
    r'F:\WikiData\ForFun\test.backward.%d.tfrecords' % config.num_sampled, config.max_question_length,
    config.num_sampled, mode='test', candidate_len=500)

  question_batch, topic_entity_batch, true_ans_batch, true_relation_batch, \
  neg_ans_batch, neg_relation_batch = tf.train.batch(
    [question, topic_entity,
     true_ans, true_relation_list,
     neg_ans, neg_relation],
    batch_size=config.batch_size,
    capacity=config.batch_size * 3 + 1000)

  bf_question_batch, bf_topic_entity_batch, bf_true_ans_batch, bf_true_relation_batch, \
  bf_neg_ans_batch, bf_neg_relation_batch = tf.train.batch(
    [bf_question, bf_topic_entity,
     bf_true_ans, bf_true_relation_list,
     bf_neg_ans, bf_neg_relation],
    batch_size=config.batch_size,
    capacity=config.batch_size * 3 + 1000)

  test_question_batch, test_topic_entity_batch, test_true_ans_batch, test_true_relation_batch, \
  test_candidate_ans_batch, test_candidate_relation_batch = tf.train.batch(
    [test_question, test_topic_entity,
     test_true_ans_list, test_true_relation_list,
     test_candidate_ans, test_candidate_realtion],
    batch_size=config.batch_size,
    capacity=config.batch_size * 3 + 1000)

  bf_test_question_batch, bf_test_topic_entity_batch, bf_test_true_ans_batch, bf_test_true_relation_batch, \
  bf_test_candidate_ans_batch, bf_test_candidate_relation_batch = tf.train.batch(
    [bf_test_question, bf_test_topic_entity,
     bf_test_true_ans_list, bf_test_true_relation_list,
     bf_test_candidate_ans, bf_test_candidate_realtion],
    batch_size=config.batch_size,
    capacity=config.batch_size * 3 + 1000)

  config.entities_vocab_size = len(item_vocab)
  config.relations_vocab_size = len(relation_vocab)
  config.words_vocab_size = len(word_vocab)

  print('Model')
  with tf.name_scope('Train'):
    with tf.variable_scope("Model", reuse=None):
      model = CNNModel(config, is_training=True)

  with tf.name_scope('Test'):
    with tf.variable_scope("Model", reuse=True):
      test_model = CNNModel(config, is_training=False, is_test=True)

  a_count = 0
  b_count = 0

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 将预训练词向量赋给模型
    model.assign_word_embedding(sess, word_embedding)
    del word_embedding

    item_embeddings, relation_embeddings = get_pretrained_entity_embeddings(item_embeddings_path,
                                                                            relation_embeddings_path)
    model.assign_relation_embedding(sess, relation_embeddings)
    del relation_embeddings

    model.assign_item_embedding(sess, item_embeddings)
    del item_embeddings

    time0 = time.time()
    for step in range(20000):
      q, topic, gt_ans, gt_relation, n_ans, n_relation = sess.run(
        [question_batch, topic_entity_batch, true_ans_batch, true_relation_batch,
         neg_ans_batch, neg_relation_batch])

      _, loss, acc = sess.run([model.cos_sim_train_op, model.cos_similarity_loss, model.accuracy],
                              {model.question_ids: q, model.topic_entity_id: topic,
                               model.true_ans: gt_ans, model.true_relation: gt_relation,
                               model.neg_ans: n_ans, model.neg_relation: n_relation,
                               model.is_forward_data: True})

      # bf_q, bf_topic, bf_gt_ans, bf_gt_relation, bf_n_ans, bf_n_relation = sess.run(
      #   [bf_question_batch, bf_topic_entity_batch, bf_true_ans_batch, bf_true_relation_batch,
      #    bf_neg_ans_batch, bf_neg_relation_batch])
      # _, bf_loss, bf_acc = sess.run([model.cos_sim_train_op,
      #                                model.cos_similarity_loss, model.accuracy],
      #                               {model.question_ids: bf_q, model.topic_entity_id: bf_topic,
      #                                model.true_ans: bf_gt_ans, model.true_relation: bf_gt_relation,
      #                                model.neg_ans: bf_n_ans, model.neg_relation: bf_n_relation,
      #                                model.is_forward_data: False})

      if step % 100 == 0:
        # q, topic, gt_ans, gt_relation, n_ans, n_relation = sess.run(
        #   [question_batch, topic_entity_batch, true_ans_batch, true_relation_batch,
        #    neg_ans_batch, neg_relation_batch])
        # _, loss, acc = sess.run([model.cos_sim_train_op, model.cos_similarity_loss, model.accuracy],
        #                         {model.question_ids: q, model.topic_entity_id: topic,
        #                          model.true_ans: gt_ans, model.true_relation: gt_relation,
        #                          model.neg_ans: n_ans, model.neg_relation: n_relation,
        #                          model.is_forward_data: True})
        # print('##step=%d##'%i,loss, acc, bf_loss, bf_acc, time.time() - time0)
        print('##step=%d##' % step)
        time0 = time.time()

        test_q, test_topic_entity, test_ans_list, test_relation_list, \
        test_c_ans, test_c_relation = sess.run([test_question_batch, test_topic_entity_batch,
                                                test_true_ans_batch, test_true_relation_batch,
                                                test_candidate_ans_batch, test_candidate_relation_batch])
        cos = sess.run(test_model.cos_sim,
                       {test_model.question_ids: test_q, test_model.topic_entity_id: test_topic_entity,
                        test_model.candidate_ans: test_c_ans, test_model.candidate_relation: test_c_relation,
                        test_model.is_forward_data: True})

        # train_cos = sess.run(test_model.cos_sim,
        #                      {test_model.question_ids: q, test_model.topic_entity_id: topic,
        #                       test_model.candidate_ans: n_ans, test_model.candidate_relation: n_relation,
        #                       test_model.is_forward_data: True})
        # cos = train_cos

        # bf_test_q, bf_test_topic_entity, bf_test_ans_list, bf_test_relation_list, \
        # bf_test_c_ans, bf_test_c_relation = sess.run([bf_test_question_batch, bf_test_topic_entity_batch,
        #                                               bf_test_true_ans_batch, bf_test_true_relation_batch,
        #                                               bf_test_candidate_ans_batch, bf_test_candidate_relation_batch])
        # bf_test_c_ans=bf_test_c_ans[:,:500]
        # bf_test_c_relation=bf_test_c_relation[:,:500]
        #
        # bf_cos = sess.run(test_model.cos_sim,
        #                   {test_model.question_ids: bf_test_q, test_model.topic_entity_id: bf_test_topic_entity,
        #                    test_model.candidate_ans: bf_test_c_ans, test_model.candidate_relation: bf_test_c_relation,
        #                    test_model.is_forward_data: False})
        # print(cos)
        for cur_b in range(config.batch_size):
          if step >= 3000:
            a_count += 1
          top_5_ans_index = heapq.nlargest(5, range(len(test_c_ans[cur_b])), cos[cur_b].take)
          top_5_ans = [test_c_ans[cur_b][index] for index in top_5_ans_index]
          ans = [a for a in test_ans_list[cur_b] if a >= 0]

          # # 在训练集上测试拟合能力的代码
          # top_5_ans_index = heapq.nlargest(5, range(len(n_ans[b])), cos[b].take)
          # top_5_ans = [n_ans[b][index] for index in top_5_ans_index]
          # ans = [gt_ans[b]]
          #
          # for aaa in ans:
          #   if aaa in n_ans[b]:
          #     print('|√|', end='\t')
          #     a_count += 1
          #     break
          for aaa in top_5_ans:
            if aaa in ans:
              print('√', end='\t')
              if step >= 3000:
                b_count += 1
              break
          if step >= 3000:
            print(b_count / a_count, '\t', ans, '\t', top_5_ans)
          else:
            print(ans, '\t', top_5_ans)


            # bf_top_5_ans_index = heapq.nlargest(5, range(len(bf_test_c_ans[i])), bf_cos[i].take)
            # bf_top_5_ans = [bf_test_c_ans[i][index] for index in bf_top_5_ans_index]
            # bf_ans = [a for a in bf_test_ans_list[i] if a >= 0]
            # for aaa in bf_top_5_ans:
            #   if aaa in bf_ans:
            #     print('√', end='\t')
            #     break
            # print(bf_ans, '\t', bf_top_5_ans)

    coord.request_stop()
    coord.join(threads)


def pred():
  pass


if __name__ == '__main__':
  main()
  # pred()
