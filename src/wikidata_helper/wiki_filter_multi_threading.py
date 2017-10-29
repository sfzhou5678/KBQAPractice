"""
# @ DeprecationWarning
  # 由于Python中GIL的存在，多线程对CPU密集型的任务没有任何帮助
  # 所以这个方法已经可以弃用了

"""
import os
import time
import json
import threading
import pickle

"""
根据QA的QID，从完整200多G的wikiJson数据中筛选数据
筛选条件：
1. 若某行数据自己的QID或其claims中涉及的QID在QA-QID中出现过，那么就将其保留
2. (暂时不考虑这一条)按0%的比例，额外随机选取数据
"""


def get_qid_set(train_file_path='../data/trains_ansEntity_fixConnectErr.txt',
                test_file_path='../data/ts_ansEntity_raw.txt',
                pkl_save_path=None):
  """
  ### 一些相关统计：
  1. 训练集+测试集的[topicEntities]总计有6041个，2491种(有3550个重复)
  2. 训练集+测试集的[ansEntities]总计有52291个，23124种(有29167个重复)
  3. 训练集+测试集的[topicEntities+ansEntities]总计有58332个，24945种(33387个重复)
  """
  if pkl_save_path is not None:
    try:
      f_read = open(pkl_save_path, 'rb')
      qid_set = pickle.load(f_read)
      return qid_set
    except:
      pass

  qid_set = set()
  reduplicated_qid_count = 0
  with open(train_file_path) as f:
    lines = f.readlines()
    for line in lines:
      data = eval(line.strip())

      if 'entities' in data['parsed_question']:
        entities = data['parsed_question']['entities']
        for entity in entities:
          if 'wikidataId' in entity:
            topic_entity_qid = entity['wikidataId']
            if topic_entity_qid in qid_set:
              reduplicated_qid_count += 1
            qid_set.add(topic_entity_qid)

      for ans in data['ans']:
        for ans_entity in ans['entities']:
          ans_qid = ans_entity['Qid']
          if ans_qid in qid_set:
            reduplicated_qid_count += 1
          qid_set.add(ans_qid)

  with open(test_file_path) as f:
    lines = f.readlines()
    for line in lines:
      data = eval(line.strip())

      if 'entities' in data['parsed_question']:
        entities = data['parsed_question']['entities']
        for entity in entities:
          if 'wikidataId' in entity:
            topic_entity_qid = entity['wikidataId']
            if topic_entity_qid in qid_set:
              reduplicated_qid_count += 1
            qid_set.add(topic_entity_qid)

      for ans in data['ans']:
        for ans_entity in ans['entities']:
          ans_qid = ans_entity['Qid']
          if ans_qid in qid_set:
            reduplicated_qid_count += 1
          qid_set.add(ans_qid)

  if pkl_save_path is not None:
    try:
      f_write = open(pkl_save_path, 'wb')
      pickle.dump(qid_set, f_write, True)
    except:
      pass
  # print(len(qid_set))
  # print(reduplicated_qid_count)

  return qid_set


def handle_data(qid_set, f_read, f_write=None):
  # @ DeprecationWarning
  # 由于Python中GIL的存在，多线程对CPU密集型的任务没有任何帮助
  # 所以这个方法已经可以弃用了
  global r_lock
  global w_lock
  global global_line_count
  global selected_line_count

  while True:
    # 1. read line

    r_lock.acquire()
    line = f_read.readline().strip()
    if line == ']' or line == '':
      # 最后一行是一个']'
      r_lock.release()
      break
    if global_line_count % 1000 == 0:
      print(global_line_count)
    r_lock.release()

    if line.endswith(','):
      line = line[:-1]
    global_line_count += 1

    # 2. handle data
    try:
      # 单线程的情况下，每处理10000W行大约花费6s，
      # 多线程的情况下，
      # time.sleep(0.01)
      # q_item = json.loads(line)
      q_item = eval(line)
      for predicate in q_item['claims']:
        revelant_entities = q_item['claims'][predicate]
        for revelant_entity in revelant_entities:
          main_snak = revelant_entity['mainsnak']
          if main_snak['datavalue']['type'] == 'wikibase-entityid':
            revelant_qid = main_snak['datavalue']['value']['id']
            if revelant_qid in qid_set:
              selected_line_count += 1
              continue
    except:
      pass

      # # 3. write result
      # w_lock.acquire()
      # f_write.write(line + '\n')
      # f_write.flush()
      # w_lock.release()

    if global_line_count >= 10000:
      break


r_lock = threading.RLock()
w_lock = threading.RLock()
global_line_count = 0
selected_line_count = 0


def main():
  data_folder = r'F:\WikiData'
  wikidata_file_path = os.path.join(data_folder, 'latest-all.json')
  selected_wikidata_path = os.path.join(data_folder, 'selected-latest-all.data')

  qid_set = get_qid_set(train_file_path='../../data/trains_ansEntity_fixConnectErr.txt',
                        test_file_path='../../data/ts_ansEntity_raw.txt',
                        pkl_save_path='../../qid_set.pkl')
  print('len(qid_set) = %d' % len(qid_set))

  f_read = open(wikidata_file_path)
  f_write = open(selected_wikidata_path, 'w')

  time0 = time.time()
  thread_num = 5
  thread_list = []
  for _ in range(thread_num):
    t = threading.Thread(target=handle_data, args=(qid_set, f_read, f_write))
    thread_list.append(t)
    t.start()

  for i in range(thread_num):
    thread = thread_list[i]
    thread.join()
  print(time.time() - time0)


if __name__ == '__main__':
  main()
