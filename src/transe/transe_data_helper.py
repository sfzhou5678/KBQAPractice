import os
import random
import time
import numpy as np
import tensorflow as tf
from collections import Counter
import collections


def filter_data():
  """
  从NZY提供的1K小文件中按需筛选出部分数据，随机排列后按8:2划分为训练集和测试集

  用only_entity模式会得到270W左右的triples
  :return:
  """
  data_folder = r'D:\DeeplearningData\NLP-DATA\英文QA\splitFB\1K'
  files_num = 1000

  mode = 'only_entity'
  # mode='all_data'

  all_data = []
  for i in range(files_num):
    file_path = os.path.join(data_folder, '%d.txt' % i)
    with open(file_path, encoding='utf-8') as f:
      lines = f.readlines()
      for line in lines:
        line = line.strip()
        head, relation, tail, _ = line.split('\t')
        if mode == 'only_entity':
          if not (tail.startswith("<") and tail.endswith('>')):
            continue
        all_data.append([head, relation, tail])

  print(len(all_data))
  random.shuffle(all_data)
  training_data = all_data[:int(len(all_data) * 0.8)]
  test_data = all_data[int(len(all_data) * 0.8):]

  trainig_test_data_folder = r'D:\DeeplearningData\NLP-DATA\英文QA\splitFB'
  training_file_path = os.path.join(trainig_test_data_folder, 'train.%d.%s.txt' % (files_num, mode))
  test_file_path = os.path.join(trainig_test_data_folder, 'test.%d.%s.txt' % (files_num, mode))

  with open(training_file_path, 'w', encoding='utf-8') as wf:
    for data in training_data:
      wf.write('\t'.join(item for item in data) + '\n')

  with open(test_file_path, 'w', encoding='utf-8') as wf:
    for data in test_data:
      wf.write('\t'.join(item for item in data) + '\n')


def build_word_dictionary(words, vocab_size, UNK='_UNK_'):
  """Process raw inputs into a dataset."""
  count = [[UNK, -1]]
  count.extend(Counter(words).most_common(vocab_size - 1))

  word2id = dict()
  for word, _ in count:
    word2id[word] = len(word2id)
  unk_count = 0
  for word in words:
    if word not in word2id:
      unk_count += 1
  count[0][1] = unk_count
  id2word = dict(zip(word2id.values(), word2id.keys()))
  return count, word2id, id2word


def build_vocab(files, entity_vocab_size, relation_vocab_size):
  entities = []
  relations = []

  for file in files:
    try:
      with open(file, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
          head, relation, tail = line.strip().split('\t')
          entities.append(head)
          entities.append(tail)
          relations.append(relation)
    except:
      print('%s不存在' % file)
  print(len(entities))  # only_entity下的entities长度有540W ，entity有120W种
  print(len(relations))  # only_entity下的relations长度有270W， 但是relation的type只有3065种

  # print(Counter(entities).most_common(100000 - 1))
  # 大致耗时8秒，所以暂时不做文件缓存了
  entities_count, entities_word2id, entities_id2word = build_word_dictionary(entities, entity_vocab_size,
                                                                             UNK='UNK_ENTITY')
  relations_count, relations_word2id, relations_id2word = build_word_dictionary(relations, relation_vocab_size,
                                                                                UNK='UNK_RELATION')

  return entities_word2id, entities_id2word, \
         relations_word2id, relations_id2word


def build_ids(raw_file_path, ids_file_path, entities_word2id, relations_word2id):
  if os.path.exists(ids_file_path):
    datas = []
    try:
      with open(ids_file_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
          data = line.strip().split('\t')
          assert len(data) == 3
          data = [int(item) for item in data]
          datas.append(data)
      return datas
    except:
      print('读取ids时发生异常')
  try:
    datas = []
    with open(raw_file_path, encoding='utf-8') as f:
      with open(ids_file_path, 'w', encoding='utf-8') as wf:
        lines = f.readlines()
        for line in lines:
          head, relation, tail = line.strip().split('\t')

          if head in entities_word2id:
            head = entities_word2id[head]
          else:
            head = 0

          if tail in entities_word2id:
            tail = entities_word2id[tail]
          else:
            tail = 0

          if relation in relations_word2id:
            relation = relations_word2id[relation]
          else:
            relation = 0

          data = [head, relation, tail]
          datas.append(data)
          wf.write('\t'.join(str(item) for item in data) + '\n')
          wf.flush()
    return datas
  except:
    print('%s不存在' % raw_file_path)


def batch_producer(ids, batch_size):
  # with tf.name_scope(None, "BatchProducer", [ids, batch_size]):
  #   raw_data = tf.convert_to_tensor(ids, name="raw_data", dtype=tf.int32)
  #   raw_data = tf.reshape(raw_data, [-1])
  #
  #   data_len = tf.size(raw_data)
  #   batch_len = (data_len // batch_size)
  #   data = tf.reshape(raw_data[0: batch_size * batch_len],
  #                     [batch_size, -1])
  #
  #   assertion = tf.assert_positive(
  #     batch_len,
  #     message="epoch_size == 0, decrease batch_size or num_steps")
  #   with tf.control_dependencies([assertion]):
  #     batch_len = tf.identity(batch_len, name="epoch_size")
  with tf.name_scope(None, "BatchProducer", [ids, batch_size]):
    raw_data = tf.convert_to_tensor(ids, name="raw_data", dtype=tf.int32)
    raw_data = tf.reshape(raw_data, [-1])

    data_len = tf.size(raw_data)
    batch_len = (data_len // batch_size)//3
    data = tf.reshape(raw_data[0: batch_size * batch_len*3],
                      [batch_size, -1])

    assertion = tf.assert_positive(
      batch_len,
      message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      batch_len = tf.identity(batch_len, name="epoch_size")


    i = tf.train.range_input_producer(batch_len, shuffle=False).dequeue()

    head = tf.strided_slice(data, [0, i*3],
                            [batch_size, i*3 + 1])
    head = tf.reshape(head, [-1])

    relation = tf.strided_slice(data, [0, i*3 + 1],
                                [batch_size, i*3 + 2])
    relation = tf.reshape(relation, [-1])

    tail = tf.strided_slice(data, [0, i*3 + 2],
                            [batch_size, i*3 + 3])
    tail = tf.reshape(tail, [batch_size, 1])

    return head, relation, tail

    # x=tf.slice(data,[0,i*3],
    #          [batch_size,3])
    #
    # return data,x
