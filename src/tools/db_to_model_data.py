import os
import json
import random
import re
import string
import time
import heapq
import tensorflow as tf
import numpy as np
from src.database.DBManager import DBManager
from src.tools.common_tools import pickle_load, pickle_dump, reverse_dict, lookup_vocab


def main():
  db = DBManager(host='192.168.1.139', port=3306, user='root', psd='1405', db='kbqa')

  train_file_path = '../../data/trains_ansEntity_fixConnectErr.txt'
  train_triples_candidate_path = '../../data/train.both.triples+candidate.txt'
  test_file_path = '../../data/ts_ansEntity_raw.txt'
  test_triples_candidate_path = '../../data/test.both.triples+candidate.txt'

  qid_to_triples(db, train_file_path, train_triples_candidate_path)
  qid_to_triples(db, test_file_path, test_triples_candidate_path)

  db.close()


def get_example(question_ids, topic_entity_id,
                true_ans, true_relation,
                neg_ans, neg_relation):
  example = tf.train.Example(features=tf.train.Features(feature={
    "question": tf.train.Feature(int64_list=tf.train.Int64List(value=question_ids)),
    "topic_entity": tf.train.Feature(int64_list=tf.train.Int64List(value=[topic_entity_id])),
    "true_ans": tf.train.Feature(int64_list=tf.train.Int64List(value=[true_ans])),
    "true_relation": tf.train.Feature(int64_list=tf.train.Int64List(value=[true_relation])),
    "neg_ans": tf.train.Feature(int64_list=tf.train.Int64List(value=neg_ans)),
    "neg_relation": tf.train.Feature(int64_list=tf.train.Int64List(value=neg_relation))
  }))

  return example


def get_test_example(question_ids, topic_entity_id,
                     true_ans_list, true_relation_list,
                     candidate_ans, candidate_relation):
  example = tf.train.Example(features=tf.train.Features(feature={
    "question": tf.train.Feature(int64_list=tf.train.Int64List(value=question_ids)),
    "topic_entity": tf.train.Feature(int64_list=tf.train.Int64List(value=[topic_entity_id])),
    "true_ans": tf.train.Feature(int64_list=tf.train.Int64List(value=true_ans_list)),
    "true_relation": tf.train.Feature(int64_list=tf.train.Int64List(value=true_relation_list)),
    "candidate_ans": tf.train.Feature(int64_list=tf.train.Int64List(value=candidate_ans)),
    "candidate_relation": tf.train.Feature(int64_list=tf.train.Int64List(value=candidate_relation)),
  }))

  return example


def get_tf_data(tf_records_path, question_max_length, num_samples, mode='train', candidate_len=500):
  # fixme: 函数名修改
  # fixme: 功能待测试
  reader = tf.TFRecordReader()
  filename_queue = tf.train.string_input_producer([tf_records_path])

  _, serialized_example = reader.read(filename_queue)

  if mode == 'train':
    features = tf.parse_single_example(
      serialized_example,
      features={
        'question': tf.FixedLenFeature([question_max_length], tf.int64),
        "topic_entity": tf.FixedLenFeature([1], tf.int64),
        'true_ans': tf.FixedLenFeature([1], tf.int64),
        'true_relation': tf.FixedLenFeature([1], tf.int64),
        'neg_ans': tf.FixedLenFeature([num_samples], tf.int64),
        'neg_relation': tf.FixedLenFeature([num_samples], tf.int64)
      })

    question = tf.cast(features['question'], tf.int32)
    topic_entity = tf.cast(features['topic_entity'], tf.int32)[0]

    true_ans = tf.cast(features['true_ans'], tf.int32)[0]
    true_relation = tf.cast(features['true_relation'], tf.int32)[0]

    neg_ans = tf.cast(features['neg_ans'], tf.int32)
    neg_relation = tf.cast(features['neg_relation'], tf.int32)

    return question, topic_entity, \
           true_ans, true_relation, \
           neg_ans, neg_relation
  elif mode == 'test':
    features = tf.parse_single_example(
      serialized_example,
      features={
        'question': tf.FixedLenFeature([question_max_length], tf.int64),
        "topic_entity": tf.FixedLenFeature([1], tf.int64),
        'true_ans': tf.FixedLenFeature([20], tf.int64),
        'true_relation': tf.FixedLenFeature([20], tf.int64),
        'candidate_ans': tf.FixedLenFeature([candidate_len], tf.int64),
        'candidate_relation': tf.FixedLenFeature([candidate_len], tf.int64)
      })

    question = tf.cast(features['question'], tf.int32)
    topic_entity = tf.cast(features['topic_entity'], tf.int32)[0]

    true_ans = tf.cast(features['true_ans'], tf.int32)
    true_relation = tf.cast(features['true_relation'], tf.int32)

    candidate_ans = tf.cast(features['candidate_ans'], tf.int32)
    candidate_relation = tf.cast(features['candidate_relation'], tf.int32)

    return question, topic_entity, \
           true_ans, true_relation, \
           candidate_ans, candidate_relation


def triples_to_tfrecords(triples_path, tfrecords_folder_path,
                         word_vocab, item_vocab, relation_vocab,
                         question_max_length,
                         padding_id,
                         num_samples=64,
                         mode='train'):
  """
  本函数的作用是将triples文件转换成tfrecords。
  
  考虑到原triples中会有多个forward_ans以及多个reverse_ans：
  1) 在tfRecords中会将每个ans单独列作一行。
  2) 每一行包括：question对应的ids；正确答案的h，r，t；补全到定长数组的负样本候选答案的[(h,r,t),(h,r,t)]
  3) 正反向数据分别写在两个文件中 
  
  :param triples_path: 
  :param tfrecords_folder_path:
  :return: 
  """
  forward_tfrecords_path = os.path.join(tfrecords_folder_path, '%s.forward.%d.tfrecords' % (mode, num_samples))
  backward_tfrecords_path = os.path.join(tfrecords_folder_path, '%s.backward.%d.tfrecords' % (mode, num_samples))

  if os.path.exists(forward_tfrecords_path) and os.path.exists(backward_tfrecords_path):
    return

  punctuation = string.punctuation
  forward_writer = tf.python_io.TFRecordWriter(forward_tfrecords_path)
  backward_writer = tf.python_io.TFRecordWriter(backward_tfrecords_path)
  with open(triples_path) as f:
    while True:
      line = f.readline().strip()
      if line == '':
        break
      datas = eval(line)
      raw_question = datas['question']
      raw_topic_entity = datas['topic_entity']

      raw_forward_ans = datas['forward_ans']
      raw_forward_candidate_ans = datas['forward_candidate_ans']

      raw_backward_ans = datas['reverse_ans']
      raw_backward_candidate_ans = datas['reverse_candidate_ans']

      # 开始处理
      raw_question = re.sub(r'[{}]+'.format(punctuation), ' ', raw_question).strip()
      question_ids = [lookup_vocab(word_vocab, word) for word in re.split('\s+', raw_question)]
      question_ids = question_ids[:question_max_length]
      for i in range(question_max_length - len(question_ids)):
        question_ids.append(padding_id)

      topic_entity_id = lookup_vocab(item_vocab, raw_topic_entity)
      # forward的第一个是topic，最后一个是ans；backward的第一个是ans，最后一个是topic
      forward_candidate_ans = [
        (lookup_vocab(item_vocab, h), lookup_vocab(relation_vocab, r), lookup_vocab(item_vocab, t))
        for (h, r, t) in raw_forward_candidate_ans]
      backward_candidate_ans = [
        (lookup_vocab(item_vocab, h), lookup_vocab(relation_vocab, r), lookup_vocab(item_vocab, t))
        for (h, r, t) in raw_backward_candidate_ans]

      if len(raw_forward_ans) > 0:
        if mode == 'train':
          n_forward = (len(forward_candidate_ans) - 1) // num_samples + 1  # 先-1再+1是为了避免len=64这类情况
          forward_need_sample = n_forward * num_samples - len(forward_candidate_ans)
          for _ in range(forward_need_sample):
            sampled_r = random.randint(1, len(relation_vocab) - 1)
            sampled_t = random.randint(1, len(item_vocab) - 1)
            forward_candidate_ans.append((topic_entity_id, sampled_r, sampled_t))

          for (h, r, t) in raw_forward_ans:
            true_relation = lookup_vocab(relation_vocab, r)
            true_ans = lookup_vocab(item_vocab, t)

            for i in range(n_forward):
              candidate_ans = forward_candidate_ans[i * num_samples:(i + 1) * num_samples]
              neg_ans = [t for (h, r, t) in candidate_ans]
              neg_relation = [r for (h, r, t) in candidate_ans]

              example = get_example(question_ids, topic_entity_id, true_ans, true_relation,
                                    neg_ans, neg_relation)
              forward_writer.write(example.SerializeToString())
        elif mode == 'test':
          # FIXME: 对于多个答案的临时做法：全都填充至20个，如果答案多余20个，就把多出来的抛弃掉
          true_ans_list = []
          true_realation_list = []
          for (h, r, t) in raw_forward_ans:
            true_relation = lookup_vocab(relation_vocab, r)
            true_ans = lookup_vocab(item_vocab, t)
            true_ans_list.append(true_ans)
            true_realation_list.append(true_relation)
            if len(true_ans_list) >= 20:
              break
          for _ in range(20 - len(raw_forward_ans)):
            true_ans_list.append(-1)
            true_realation_list.append(-1)

          # fixme: 由于fixedLengthFeatures的存在，临时将candidate填充至500
          candidate_ans = forward_candidate_ans[:500]
          candidate_a = [t for (h, r, t) in candidate_ans]
          candidate_r = [r for (h, r, t) in candidate_ans]
          for _ in range(500 - len(candidate_a)):
            candidate_a.append(0)
            candidate_r.append(0)

          example = get_test_example(question_ids, topic_entity_id, true_ans_list, true_realation_list,
                                     candidate_a, candidate_r)
          forward_writer.write(example.SerializeToString())
      if len(raw_backward_ans) > 0:
        if mode == 'train':
          n_backward = (len(backward_candidate_ans) - 1) // num_samples + 1  # 先-1再+1是为了避免len=64这类情况
          backward_need_sample = n_backward * num_samples - len(backward_candidate_ans)
          for _ in range(backward_need_sample):
            sampled_r = random.randint(1, len(relation_vocab) - 1)
            sampled_t = random.randint(1, len(item_vocab) - 1)
            backward_candidate_ans.append((sampled_t, sampled_r, topic_entity_id))  # 注意，这里是topic在最後，与forward不同

          for (h, r, t) in raw_backward_ans:
            true_relation = lookup_vocab(relation_vocab, r)
            true_ans = lookup_vocab(item_vocab, h)  # 注意，这里是h而不是t，与forward不同

            for i in range(n_backward):
              candidate_ans = backward_candidate_ans[i * num_samples:(i + 1) * num_samples]
              neg_ans = [h for (h, r, t) in candidate_ans]  # 注意，这里是h而不是t，与forward不同
              neg_relation = [r for (h, r, t) in candidate_ans]

              example = get_example(question_ids, topic_entity_id, true_ans, true_relation,
                                    neg_ans, neg_relation)
              backward_writer.write(example.SerializeToString())
        elif mode == 'test':
          # FIXME: 对于多个答案的临时做法：全都填充至20个，如果答案多余20个，就把多出来的抛弃掉
          true_ans_list = []
          true_realation_list = []
          for (h, r, t) in raw_backward_ans:
            true_relation = lookup_vocab(relation_vocab, r)
            true_ans = lookup_vocab(item_vocab, h)  # 注意这里是h不是t
            true_ans_list.append(true_ans)
            true_realation_list.append(true_relation)
            if len(true_ans_list) >= 20:
              break
          for _ in range(20 - len(raw_backward_ans)):
            true_ans_list.append(-1)
            true_realation_list.append(-1)

          # fixme: 由于fixedLengthFeatures的存在，临时将candidate填充至500
          candidate_ans = backward_candidate_ans[:500]
          candidate_a = [h for (h, r, t) in candidate_ans]  # 注意是h不是t
          candidate_r = [r for (h, r, t) in candidate_ans]
          for _ in range(500 - len(candidate_a)):
            candidate_a.append(0)
            candidate_r.append(0)

          example = get_test_example(question_ids, topic_entity_id, true_ans_list, true_realation_list,
                                     candidate_a, candidate_r)
          backward_writer.write(example.SerializeToString())
  forward_writer.close()
  backward_writer.close()


def qid_to_triples(db, file_path, save_path):
  """
  本函数作用是通过查询DB：
  1) 将原本用Topic、ansQid表示的问答数据集转化成topic+relation=ansQid或ansQid+relation=topic的三元组
  2) 筛选出每个topic的正向以及反向的候选答案集
  3) 将所得的结果记录下来
  
  最终记录的每行的数据大致为：
  {"question":"xxxx", "topic_entity":"QXXX",
  "forward_ans":[(q1,r,q2)],"forward_candidate_ans":[(q1,r,q2),(q1,r,q2)],
  "reverse_ans":[(q1,r,q2)],"reverse_candidate_ans":[(q1,r,q2),(q1,r,q2)]
  }
  # 弃用版本：
  "forward_ans":[{"head":"Q1","relation":"P","tail":"Q2"}],"forward_candidate_ans":[{"head":"Q3","relation":"P","tail":"Q4"}],
  "reverse_ans":[{"head":"Q5","relation":"P","tail":"Q6"}],"reverse_candidate_ans":[{"head":"Q7","relation":"P","tail":"Q8"}],
  :param file_path: 
  :return: 
  """
  line_count = 0
  no_ans_count = 0
  with open(save_path, 'w') as wf:
    with open(file_path) as f:
      lines = f.readlines()
      for line in lines:
        data = eval(line.strip())
        if 'entities' in data['parsed_question']:
          line_count += 1
          entities = data['parsed_question']['entities']
          for entity in entities:
            if 'wikidataId' in entity:
              topic_entity_qid = entity['wikidataId']
              # 每个问题会对应多个答案
              # 而且每个text的ans会找到多个对应的Qid(比如apple)
              # 经过测试发现：一个text对应的通常一组QID中能正确匹配的只有0-1个，所以可以不用考虑消歧的问题

              # print(data['question'], entity['entityId'],[ans['name'] for ans in data['ans']])

              forward_candidate_ans = list(db.select_from_topic(topic_entity_qid, max_depth=1))
              reverse_candidate_ans = list(db.select_to_topic(topic_entity_qid, max_depth=1))

              forward_ans = []
              reverse_ans = []
              for ans in data['ans']:
                for ans_entity in ans['entities']:
                  ans_qid = ans_entity['Qid']

                  # 1. 正向匹配
                  relevant_triples = db.select_by_head_and_tail(topic_entity_qid, ans_qid)
                  if len(relevant_triples) > 0:
                    # print('√',ans_entity['label'],',',ans_entity['description'])

                    for (topic_qid, r, ans_qid) in relevant_triples:
                      # 在这种情况下，h=topic, t=ans
                      ans = (topic_qid, r, ans_qid)
                      forward_ans.append(ans)

                  # 2. 逆向匹配
                  reversed_relevant_triples = db.select_by_head_and_tail(ans_qid, topic_entity_qid, )
                  if len(reversed_relevant_triples) > 0:
                    # print('R√',ans_entity['label'],',',ans_entity['description'])
                    for (ans_qid, r, topic_qid) in reversed_relevant_triples:
                      # 在这种情况下，h=ans，t=topic
                      ans = (ans_qid, r, topic_qid)
                      reverse_ans.append(ans)

              if len(forward_ans) == 0 and len(reverse_ans) == 0:
                no_ans_count += 1
                continue
              dict_to_write = {}
              dict_to_write['question'] = data['question']
              dict_to_write['topic_entity'] = topic_entity_qid

              dict_to_write['forward_ans'] = forward_ans
              random.shuffle(forward_candidate_ans)
              dict_to_write['forward_candidate_ans'] = forward_candidate_ans

              if len(reverse_candidate_ans) > 10000:
                reverse_candidate_ans = random.sample(reverse_candidate_ans, 10000 - len(reverse_ans))
                for ans in reverse_ans:
                  reverse_candidate_ans.append(ans)

              dict_to_write['reverse_ans'] = reverse_ans
              random.shuffle(reverse_candidate_ans)
              dict_to_write['reverse_candidate_ans'] = reverse_candidate_ans

              str_to_write = str(dict_to_write)
              # str_to_write=json.dumps(dict_to_write)
              wf.write(str_to_write + '\n')
              # wf.flush()

          if line_count % 100 == 0:
            print(line_count, no_ans_count)
      print(line_count, no_ans_count)


def prepare_transe_data(selected_wikidata_file_path, transe_data_save_path):
  total_lines = 0
  with open(transe_data_save_path, 'w') as wf:
    with open(selected_wikidata_file_path) as f:
      while True:
        line = f.readline().strip()
        if line == '':
          break
        q_item = json.loads(line)
        QID = q_item['id']

        for predicate in q_item['claims']:
          relevant_entities = q_item['claims'][predicate]
          for revelant_entity in relevant_entities:
            try:
              main_snak = revelant_entity['mainsnak']
              relevant_qid = main_snak['datavalue']['value']['id']

              triple_to_write = (QID, predicate, relevant_qid)
              wf.write('\t'.join(item for item in triple_to_write) + '\n')
              total_lines += 1
              if total_lines % 1000000 == 0:
                print(total_lines)
            except:
              pass
  print(total_lines)


def build_entities_counter(transe_data_save_path, item_counter_save_path, relation_counter_save_path):
  def add_to_dic(dic, data):
    if data not in dic:
      dic[data] = 0
    dic[data] += 1

  item_word_counter = {}
  relation_word_counter = {}

  total_count = 0
  with open(transe_data_save_path) as f:
    while True:
      line = f.readline().strip()
      if line == '':
        break

      total_count += 1
      if total_count % 1000000 == 0:
        print(total_count)

        print(len(item_word_counter))
        # print(item_word_counter)

        print(len(relation_word_counter))
        # print(relation_word_counter)

      head, relation, tail = line.split('\t')
      add_to_dic(item_word_counter, head)
      add_to_dic(item_word_counter, tail)

      add_to_dic(relation_word_counter, relation)

  print(len(item_word_counter))

  print(len(relation_word_counter))
  pickle_dump(item_word_counter, item_counter_save_path)
  pickle_dump(relation_word_counter, relation_counter_save_path)


def get_entity_vocabulary(counter_path, vocab_save_path, UNK='_UNK_', percent=1.0):
  if os.path.exists(vocab_save_path):
    try:
      vocab = pickle_load(vocab_save_path)

      return vocab
    except:
      print('读取字典%s时出错' % vocab_save_path)

  # 不存在缓存，开始制作词汇表
  counter = pickle_load(counter_path)  # dict格式
  counter = sorted(counter.items(), key=lambda d: d[1], reverse=True)  # list格式
  print(len(counter))

  vocab = {}
  vocab[UNK] = 0
  id = 1

  f_sum = sum([v for k, v in counter])
  print(f_sum)
  percent_sum = (percent * f_sum)
  cur_f = 0
  for word, frequency in counter:
    cur_f += frequency
    vocab[word] = id
    id += 1
    if cur_f > percent_sum:
      break
  print(len(vocab))

  pickle_dump(vocab, vocab_save_path)
  return vocab


def get_word_vocabulary(pretrained_wordvec_save_path, word_vocab_save_path, word_embedding_save_path,
                        UNK='WORD_UNK', PAD='PAD',
                        vocab_size=50000):
  if os.path.exists(word_vocab_save_path):
    try:
      word_vocab = pickle_load(word_vocab_save_path)
      word_embeddings = pickle_load(word_embedding_save_path)

      return word_vocab, word_embeddings
    except:
      print('读取字典%s时出错' % word_vocab_save_path)

  embeddings = []
  word_vocab = {}
  word_vocab[UNK] = 0
  embeddings.append(np.zeros([100], dtype=np.float32))
  word_vocab[PAD] = 1
  embeddings.append(np.ones([100], dtype=np.float32))

  id = 2
  with open(pretrained_wordvec_save_path, encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip()
      data = line.split(' ')
      word = data[0]
      # TODO: 是否要直接把embedding的内容return？
      embedding = data[1:]
      embedding = np.array([float(e) for e in embedding])
      embeddings.append(embedding)

      word_vocab[word] = id
      id += 1
      if id >= vocab_size:
        break
  embeddings = np.array(embeddings)
  pickle_dump(word_vocab, word_vocab_save_path)
  pickle_dump(embeddings, word_embedding_save_path)

  return word_vocab, embeddings


def get_pretrained_entity_embeddings(item_embeddings_path, relation_embeddings_path):
  """
  注意item_embeddings使用的不是pickLoad而是np.load
  这是一个特例
  :param item_embeddings_path:
  :param relation_embeddings_path:
  :return:
  """
  item_embeddings = np.load(item_embeddings_path)
  relation_embeddings = pickle_load(relation_embeddings_path)

  return item_embeddings, relation_embeddings
  # return relation_embeddings


if __name__ == '__main__':
  pass
