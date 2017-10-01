import os
import pickle
import re
import random

from src.tools.common_tools import pickle_load, pickle_dump

data_folder = r'F:\FBData'
file_path = os.path.join(data_folder, 'SmallBase_gama')
selected_data_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.se')
selected_twice_data_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.twice.se')

selected_ids_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.id.pkl')
selected_train_ids_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.train.id.pkl')
selected_test_ids_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.test.id.pkl')

total_entity_counter_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.wordCounter.total.pkl')
useful_entity_counter_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.wordCounter.useful.pkl')
relation_counter_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.relationCounter.pkl')

entity_vocab_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.entityVocab.pkl')
relation_vocab_path = os.path.join(data_folder, 'fb.wikiMappings.webquestion+random12.relationVocab.pkl')

mapping_path = os.path.join(data_folder, 'fb2w.nt')


def filter_data(freebase_mid_set):
  if not os.path.exists(file_path):
    print('文件不存在')

  entity_word_counter = {}
  relation_word_counter = {}

  def add_to_dic(dic, data):
    if data not in dic:
      dic[data] = 0
    dic[data] += 1

  with open(file_path, encoding='utf-8') as f:
    with open(selected_data_path, 'w', encoding='utf-8') as wf:
      total_count = 0
      selected_lines_count = 0
      error_count = 0
      from_fb_set_lines_count = 0
      while True:
        line = f.readline().strip()
        if line == '':
          break

        total_count += 1
        if total_count % 1000000 == 0:
          print(total_count, selected_lines_count, from_fb_set_lines_count, line.strip())
          wf.flush()

        try:
          head, relation, tail, _ = line.split('\t')
          if (not tail.startswith('<')) or (not tail.endswith('>')):
            tail = re.findall(r'".+"', tail)[0]

          if (head not in freebase_mid_set) and (tail not in freebase_mid_set):
            if total_count % 12 != 0:
              continue
          else:
            from_fb_set_lines_count += 1
          add_to_dic(entity_word_counter, head)
          add_to_dic(entity_word_counter, tail)

          add_to_dic(relation_word_counter, relation)
          wf.write('%s\t%s\t%s' % (head, relation, tail) + '\n')
          selected_lines_count += 1
        except:
          error_count += 1
          print('error %d:%s' % (error_count, line))

      print('total_count=%d, selected lines=%d, fromFBSet lines=%d' % (
        total_count, selected_lines_count, from_fb_set_lines_count))

  pickle_dump(entity_word_counter, total_entity_counter_path)
  pickle_dump(relation_word_counter, relation_counter_path)

  useful_entity_word_counter = {}
  for k in entity_word_counter:
    if entity_word_counter[k] >= 3:
      useful_entity_word_counter[k] = entity_word_counter[k]

  # useful_entity_word_counter = sorted(useful_entity_word_counter.items(), key=lambda d: d[1], reverse=True)
  # relation_word_counter = sorted(relation_word_counter.items(), key=lambda d: d[1], reverse=True)
  print(len(entity_word_counter))
  print(len(useful_entity_word_counter))
  print(len(relation_word_counter))

  pickle_dump(useful_entity_word_counter, useful_entity_counter_path)


def get_vocabulary(vocab_path, counter_path, UNK='_UNK_',percent=1.0):
  if os.path.exists(vocab_path):
    try:
      vocab = pickle_load(vocab_path)

      return vocab
    except:
      print('读取字典%s时出错' % vocab_path)

  # 不存在缓存，开始制作词汇表
  counter = pickle_load(counter_path)  # dict格式
  counter = sorted(counter.items(), key=lambda d: d[1], reverse=True)  # list格式
  print(len(counter))

  vocab = {}
  vocab[UNK]=0
  id = 1

  f_sum = sum([v for k, v in counter])
  print(f_sum)
  percent_sum = (percent * f_sum)
  cur_f = 0
  for k, v in counter:
    cur_f += v
    vocab[k] = id
    id += 1
    if cur_f > percent_sum:
      break
  print(len(vocab))

  pickle_dump(vocab, vocab_path)
  return vocab


def get_id_data(raw_file_path, ids_file_path, entities_vocab, relations_vocab):
  """
  根据vocab，将rawData转换成ids
  :param raw_file_path:
  :param ids_file_path:
  :param entities_vocab:
  :param relations_vocab:
  :return:
  """
  if os.path.exists(ids_file_path):
    try:
      datas = pickle_load(ids_file_path)

      return datas
    except:
      print('读取ids时发生异常')

  # 没有id缓存，开始制作
  datas = []
  with open(raw_file_path, encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
      head, relation, tail = line.strip().split('\t')

      if head in entities_vocab:
        head = entities_vocab[head]
      else:
        head = 0

      if tail in entities_vocab:
        tail = entities_vocab[tail]
      else:
        tail = 0

      if relation in relations_vocab:
        relation = relations_vocab[relation]
      else:
        relation = 0

      data = [head, relation, tail]
      datas.append(data)
    f_write = open(ids_file_path, 'wb')
    pickle.dump(datas, f_write, True)
  return datas


def get_train_test_ids(selected_data_path, selected_ids_path,
                       selected_train_ids_path, selected_test_ids_path,
                       entity_vocab, relation_vocab):
  """
  获取train、test的id数据
  :return:
  """
  if os.path.exists(selected_test_ids_path) and os.path.exists(selected_train_ids_path):
    try:
      train_ids = pickle_load(selected_train_ids_path)
      test_ids = pickle_load(selected_test_ids_path)
      return train_ids, test_ids
    except:
      pass
  selected_ids = get_id_data(selected_data_path,
                             ids_file_path=selected_ids_path,
                             entities_vocab=entity_vocab, relations_vocab=relation_vocab)
  random.shuffle(selected_ids)

  train_ids = selected_ids[:int(len(selected_ids) * 0.8)]
  test_ids = selected_ids[int(len(selected_ids) * 0.8):]

  pickle_dump(train_ids, selected_train_ids_path)
  pickle_dump(test_ids, selected_test_ids_path)

  return train_ids, test_ids


def filter_data2(entity_vocab):
  """
  用filter_data得到10%左右的数据和非低频词(>=3)的词汇表后之后，再根据该词汇表重新筛选一遍数据
  [代码需要整合]
  :return:
  """
  with open(selected_data_path, encoding='utf-8') as f:
    with open(selected_twice_data_path, 'w', encoding='utf-8') as wf:
      total_count = 0
      useful_count = 0
      error_count = 0
      while True:
        line = f.readline().strip()
        if line == '':
          break

        total_count += 1
        if total_count % 1000000 == 0:
          print(total_count, line.strip())
          wf.flush()

        try:
          head, relation, tail = line.split('\t')

          if (head not in entity_vocab) or (tail not in entity_vocab):
            continue
          useful_count += 1
          wf.write(line + '\n')
        except:
          error_count += 1
          print('error %d:%s' % (error_count, line))
      print(total_count, useful_count)


if __name__ == '__main__':
  # mid_read = open('data/WebQuestion.mid.pkl', 'rb')
  # freebase_mid_set = pickle.load(mid_read)
  # filter_data(freebase_mid_set)


  # filter_data2(pickle_load(useful_entity_counter_path))

  relation_vocab = get_vocabulary(relation_vocab_path, relation_counter_path, percent=1.0)
  entity_vocab = get_vocabulary(entity_vocab_path, useful_entity_counter_path, percent=0.95)

  train_ids, test_ids = get_train_test_ids(selected_twice_data_path, selected_ids_path,
                                           selected_train_ids_path, selected_test_ids_path,
                                           entity_vocab, relation_vocab)
  print(len(test_ids))
  print(test_ids[:10])
