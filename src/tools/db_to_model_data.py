import os
import json
import random
import re
import string
import tensorflow as tf
from src.database.DBManager import DBManager
from src.tools.common_tools import pickle_load, pickle_dump, reverse_dict, get_id


def main():
  db = DBManager(host='192.168.1.139', port=3306, user='root', psd='1405', db='kbqa')

  train_file_path = '../../data/trains_ansEntity_fixConnectErr.txt'
  train_triples_candidate_path = '../../data/train.both.triples+candidate.txt'
  test_file_path = '../../data/ts_ansEntity_raw.txt'
  test_triples_candidate_path = '../../data/test.both.triples+candidate.txt'

  qid_to_triples(db, train_file_path, train_triples_candidate_path)
  qid_to_triples(db, test_file_path, test_triples_candidate_path)

  db.close()


def get_example(question_ids, topic_entity_id, ans, candidate_ans):
  example = tf.train.Example(features=tf.train.Features(feature={
    "question": tf.train.Feature(int64_list=tf.train.Int64List(value=question_ids)),
    "topic_entity": tf.train.Feature(int64_list=tf.train.Int64List(value=[topic_entity_id])),
    "ans": tf.train.Feature(int64_list=tf.train.Int64List(value=ans)),
    "candidate_ans": tf.train.Feature(int64_list=tf.train.Int64List(value=candidate_ans))
  }))

  return example


def get_tf_data(tf_records_path):
  # fixme: 函数名修改
  # fixme: 功能待测试
  reader = tf.TFRecordReader()
  filename_queue = tf.train.string_input_producer([tf_records_path])

  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'question': tf.FixedLenFeature([50], tf.int64),
      "topic_entity": tf.FixedLenFeature([1], tf.int64),
      'ans': tf.FixedLenFeature([3], tf.int64),
      "candidate_ans": tf.FixedLenFeature([3 * 64], tf.int64),
    })

  question = tf.cast(features['question'], tf.int32)
  topic_entity = tf.cast(features['topic_entity'], tf.int32)
  ans = tf.cast(features['ans'], tf.int32)
  candidate_ans = tf.cast(features['candidate_ans'], tf.int32)

  return question, topic_entity, ans, candidate_ans


def triples_to_tfrecords(triples_path, tfrecords_folder_path,
                         word_vocab, item_vocab, realtion_vocab,
                         num_samples=64):
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
  forward_tfrecords_path = os.path.join(tfrecords_folder_path, 'train.forward.tfrecords')
  reverse_tfrecords_path = os.path.join(tfrecords_folder_path, 'train.reverse.tfrecords')

  count = 0

  punctuation = string.punctuation

  writer = tf.python_io.TFRecordWriter(forward_tfrecords_path)
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

      raw_reverse_ans = datas['reverse_ans']
      raw_reverse_candidate_ans = datas['reverse_candidate_ans']

      # 开始处理
      raw_question = re.sub(r'[{}]+'.format(punctuation), ' ', raw_question).strip()
      question_ids = [get_id(word_vocab, word) for word in re.split('\s+', raw_question)]
      question_ids = question_ids[:50]
      for i in range(50 - len(question_ids)):
        question_ids.append(0)

      topic_entity_id = get_id(item_vocab, raw_topic_entity)
      forward_candidate_ans = [(get_id(item_vocab, h), get_id(relation_vocab, r), get_id(item_vocab, t))
                               for (h, r, t) in raw_forward_candidate_ans]
      reverse_candidate_ans = [(get_id(item_vocab, h), get_id(relation_vocab, r), get_id(item_vocab, t))
                               for (h, r, t) in raw_reverse_candidate_ans]

      tf_data = {}
      tf_data['question'] = question_ids
      tf_data['topic_id'] = topic_entity_id

      if len(raw_forward_ans) > 0:
        n_forward = (len(forward_candidate_ans) - 1) // num_samples + 1  # 先-1再+1是为了避免len=64这类情况
        need_sample = n_forward * num_samples - len(forward_candidate_ans)
        for _ in range(need_sample):
          sampled_r = random.randint(1, len(realtion_vocab) - 1)
          sampled_t = random.randint(1, len(item_vocab) - 1)
          forward_candidate_ans.append((topic_entity_id, sampled_r, sampled_t))

        # 由于tfrecords中的Int64List必须接受list形式的数据，所以吧三元组全都拆分开来一个个排列成list
        new_forward_candidate_ans = []
        for (h, r, t) in forward_candidate_ans:
          new_forward_candidate_ans.append(h)
          new_forward_candidate_ans.append(r)
          new_forward_candidate_ans.append(t)
        forward_candidate_ans = new_forward_candidate_ans

        for (h, r, t) in raw_forward_ans:
          h_id = get_id(item_vocab, h)
          r_id = get_id(relation_vocab, r)
          t_id = get_id(item_vocab, t)
          tf_data['ans'] = [h_id, r_id, t_id]

          for i in range(n_forward):
            candidate_ans = forward_candidate_ans[i * 3 * num_samples:(i + 1) * 3 * num_samples]
            tf_data['forward_candidate_ans'] = candidate_ans
            example = get_example(question_ids, topic_entity_id, tf_data['ans'], candidate_ans)
            writer.write(example.SerializeToString())
            count += 1
        print(count)
        # del tf_data['forward_candidate_ans']  # 在reverse文件中不需要上面的forward
        # if len(raw_reverse_ans) > 0:
        #   n_reverse = (len(reverse_candidate_ans) - 1) // num_samples + 1  # 先-1再+1是为了避免len=64这类情况
        #   need_sample = n_reverse * num_samples - len(reverse_candidate_ans)
        #   for _ in range(need_sample):
        #     sampled_r = random.randint(1, len(realtion_vocab) - 1)
        #     sampled_t = random.randint(1, len(item_vocab) - 1)
        #     reverse_candidate_ans.append((sampled_t, sampled_r, topic_entity_id))
        #
        #   for (h, r, t) in raw_reverse_ans:
        #     h_id = get_id(item_vocab, h)
        #     r_id = get_id(relation_vocab, r)
        #     t_id = get_id(item_vocab, t)
        #     tf_data['ans'] = (h_id, r_id, t_id)
        #
        #     for i in range(n_reverse):
        #       candidate_ans = reverse_candidate_ans[i * num_samples:(i + 1) * num_samples]
        #       tf_data['reverse_candidate_ans'] = candidate_ans
  writer.close()


def qid_to_triples(db, file_path, saving_path):
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
  with open(saving_path, 'w') as wf:
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
            break
      print(line_count, no_ans_count)


def prepare_transe_data(selected_wikidata_file_path, transe_data_saving_path):
  total_lines = 0
  with open(transe_data_saving_path, 'w') as wf:
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


def build_entities_counter(transe_data_saving_path, item_counter_saving_path, relation_counter_saving_path):
  def add_to_dic(dic, data):
    if data not in dic:
      dic[data] = 0
    dic[data] += 1

  item_word_counter = {}
  relation_word_counter = {}

  total_count = 0
  with open(transe_data_saving_path) as f:
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
  pickle_dump(item_word_counter, item_counter_saving_path)
  pickle_dump(relation_word_counter, relation_counter_saving_path)


def get_entity_vocabulary(counter_path, vocab_saving_path, UNK='_UNK_', percent=1.0):
  if os.path.exists(vocab_saving_path):
    try:
      vocab = pickle_load(vocab_saving_path)

      return vocab
    except:
      print('读取字典%s时出错' % vocab_saving_path)

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

  pickle_dump(vocab, vocab_saving_path)
  return vocab


def get_word_vocabulary(pretrained_wordvec_saving_path, word_vocab_saving_path, UNK='WORD_UNK'):
  if os.path.exists(word_vocab_saving_path):
    try:
      word_vocab = pickle_load(word_vocab_saving_path)

      return word_vocab
    except:
      print('读取字典%s时出错' % word_vocab_saving_path)

  word_vocab = {}
  word_vocab[UNK] = 0
  id = 1
  with open(pretrained_wordvec_saving_path, encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip()
      data = line.split(' ')
      word = data[0]
      # TODO: 是否要直接把embedding的内容return？
      # embedding = data[1:]

      word_vocab[word] = id
      id += 1
  pickle_dump(word_vocab, word_vocab_saving_path)

  return word_vocab


if __name__ == '__main__':
  # 首先在main中利用DB将原始的问答对转化成Model所需的{question, topicEntity，trueAns，candidateAns}

  wikidata_folder = r'F:\WikiData'
  selected_wikidata_file_path = os.path.join(wikidata_folder, 'selected-latest-all.OnlyRelevant.data')

  transe_data_saving_path = os.path.join(wikidata_folder, 'transE.OnlyRelevant.data')
  item_counter_saving_path = os.path.join(wikidata_folder, 'item.counter.OnlyRelevant.cnt')
  relation_counter_saving_path = os.path.join(wikidata_folder, 'relation.counter.OnlyRelevant.cnt')

  item_vocab_saving_path = os.path.join(wikidata_folder, 'item.vocab.OnlyRelevant.voc')
  relation_vocab_saving_path = os.path.join(wikidata_folder, 'relation.vocab.OnlyRelevant.voc')

  pretrained_wordvec_saving_path = os.path.join(wikidata_folder, 'glove.6B.100d.txt')
  word_vocab_saving_path = os.path.join(wikidata_folder, 'word.vocab.glove.100.voc')

  # 首先在main中利用DB将原始的问答对转化成Model所需的{question, topicEntity，trueAns，candidateAns}
  # db = DBManager(host='192.168.1.139', port=3306, user='root', psd='1405', db='kbqa')
  #
  # train_file_path = '../../data/trains_ansEntity_fixConnectErr.txt'
  # train_triples_candidate_path = '../../data/train.both.triples+candidate.txt'
  # test_file_path = '../../data/ts_ansEntity_raw.txt'
  # test_triples_candidate_path = '../../data/test.both.triples+candidate.txt'
  #
  # qid_to_triples(db, train_file_path, train_triples_candidate_path)
  # qid_to_triples(db, test_file_path, test_triples_candidate_path)
  #
  # db.close()

  # 从原30G文件中提取三元组,总共提到了2300W多对
  # prepare_transe_data(selected_wikidata_file_path, transe_data_saving_path)

  # 统计三元组中item和relation的词频
  # build_entities_counter(transe_data_saving_path, item_counter_saving_path, relation_counter_saving_path)


  # 构建entities的词汇表，分别列在item和relation两个表中
  # 每个词汇表的0号位都是UNK
  relation_vocab = get_entity_vocabulary(relation_counter_saving_path, relation_vocab_saving_path, UNK='RELATION_UNK',
                                         percent=1.0)

  item_vocab = get_entity_vocabulary(item_counter_saving_path, item_vocab_saving_path, UNK='ITEM_UNK', percent=0.80)

  word_vocab = get_word_vocabulary(pretrained_wordvec_saving_path, word_vocab_saving_path, UNK='WORD_UNK')

  # 然后配合各vocab，将rawModelData转化成真正的ModelData
  # triples_to_tfrecords(r'F:\WikiData\ForFun\train.both.triples+candidate.txt', r'F:\WikiData\ForFun',
  #                      word_vocab, item_vocab, relation_vocab)

  question, topic_entity, ans, candidate_ans = get_tf_data(r'F:\WikiData\ForFun\train.forward.tfrecords')
  question_batch, topic_entity_batch, ans_batch, candidate_ans_batch = tf.train.batch(
    [question, topic_entity, ans, candidate_ans],
    batch_size=4,
    capacity=4 * 3 + 1000)

  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(80):
      q, t, a, ca = sess.run([question_batch, topic_entity_batch, ans_batch, candidate_ans_batch])
      print(t)

    coord.request_stop()
    coord.join(threads)
