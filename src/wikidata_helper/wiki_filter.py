import os
import time
import json
import pickle

"""
根据QA的QID，从完整200多G的wikiJson数据中筛选数据
筛选条件：
1. 若某行数据自己的QID或其claims中涉及的QID在QA-QID中出现过，那么就将其保留
2. (暂时不考虑这一条)按0%的比例，额外随机选取数据
"""


def get_qid_set(train_file_path='../data/trains_ansEntity_fixConnectErr.txt',
                test_file_path='../data/ts_ansEntity_raw.txt',
                pkl_saving_path=None):
  """
  ### 一些相关统计：
  1. 训练集+测试集的[topicEntities]总计有6041个，2491种(有3550个重复)
  2. 训练集+测试集的[ansEntities]总计有52291个，23124种(有29167个重复)
  3. 训练集+测试集的[topicEntities+ansEntities]总计有58332个，24945种(33387个重复)
  """
  if pkl_saving_path is not None:
    try:
      f_read = open(pkl_saving_path, 'rb')
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

  if pkl_saving_path is not None:
    try:
      f_write = open(pkl_saving_path, 'wb')
      pickle.dump(qid_set, f_write, True)
    except:
      pass
  # print(len(qid_set))
  # print(reduplicated_qid_count)

  return qid_set


def select_relevant_entities(qid_set, wikidata_file_path, selected_data_path):
  """
  筛选规则：
  1. 对于每一行数据，若这个entity本身位于qidSet中，就将这一整行数据保存下来
  2. 对于所有与该entity有关联的实体，只将QID位于qidSet中的关系对记录下来，其余的数据抛弃

  功效：相对于整行选取，这个方法可以将关系对的数量减少到20%左右，占用空间压缩到30%左右

  数据情况：
  # 260G的json文件大致3590W行
  # 不写数据的情况下每处理10W行大约花费55s，
  # 写数据的情况下每10W行大约??s
  # 前10W行数据中包含28W对关系
  # 前30W行数据中包含64W对关系
  """
  line_count = 0
  selected_line_count = 0
  total_triple_count = 0
  error_count = 0

  with open(selected_data_path, 'w') as wf:
    with open(wikidata_file_path) as f:
      f.readline()  # 由于第一行是一个'[' ,所以把他跳过
      time0 = time.time()
      while True:
        line = f.readline().strip()
        if line == ']' or line == '':
          # 最后一行是一个']'
          break

        if line.endswith(','):
          line = line[:-1]

        need_select = False  # 这个needSelect是针对某一行的，记录这一行数据是否需要筛选出来
        q_item = json.loads(line)
        QID = q_item['id']

        if QID in qid_set:
          total_triple_count += sum([len(q_item['claims'][predicate]) for predicate in q_item['claims']])
          data_to_write = line
          need_select = True
        else:
          claims = {}
          for predicate in q_item['claims']:
            relevant_entities = q_item['claims'][predicate]
            relations = []
            for revelant_entity in relevant_entities:
              try:
                main_snak = revelant_entity['mainsnak']
                if main_snak['datavalue']['type'] == 'wikibase-entityid':
                  revelant_qid = main_snak['datavalue']['value']['id']
                  if revelant_qid in qid_set:
                    total_triple_count += 1
                    relations.append(revelant_entity)
                    need_select = True
              except:
                error_count += 1
                pass
            if len(relations) > 0:
              claims[predicate]=relations
          q_item['claims'] = claims
          data_to_write = json.dumps(q_item)

        line_count += 1
        if need_select:
          selected_line_count += 1
          wf.write(data_to_write + '\n')
          pass
        if line_count % 100000 == 0:
          print('[line]:%d [selectedLine]:%d [triples]:%d [timeConsumed]:%ds' % (
            line_count, selected_line_count, total_triple_count, time.time() - time0))
          time0 = time.time()


def select_the_whole_line(qid_set, wikidata_file_path, selected_data_path):
  """
  筛选规则：
  1. 对于每一行数据，只要这个entity本身或任意一个有关联的mainsnak的QID位于qidSet中，就将这一整行数据保存下来

  数据情况：
  # 260G的json文件大致3590W行
  # 不写数据的情况下每处理10W行大约花费40s，
  # 写数据的情况下每10W行大约70s
  # 前10W行数据中需筛选出6W行
  # 前30W行数据中需筛选出19W行
  # 前100W行中需筛选出58W行

  # 前10W行数据中包含154W对关系
  # 前30W行数据中包含343W对关系
  """
  line_count = 0
  selected_line_count = 0
  total_triple_count = 0
  error_count = 0

  with open(selected_data_path, 'w') as wf:
    with open(wikidata_file_path) as f:
      f.readline()  # 由于第一行是一个'[' ,所以把他跳过
      time0 = time.time()
      while True:
        line = f.readline().strip()
        if line == ']' or line == '':
          # 最后一行是一个']'
          break

        if line.endswith(','):
          line = line[:-1]

        need_select = False  # 这个needSelect是针对某一行的，记录这一行数据是否需要筛选出来
        q_item = json.loads(line)
        QID = q_item['id']
        if QID in qid_set:
          need_select = True
        for predicate in q_item['claims']:
          if need_select:
            break
          revelant_entities = q_item['claims'][predicate]
          for revelant_entity in revelant_entities:
            try:
              main_snak = revelant_entity['mainsnak']
              if main_snak['datavalue']['type'] == 'wikibase-entityid':
                revelant_qid = main_snak['datavalue']['value']['id']
                if revelant_qid in qid_set:
                  need_select = True
                  break
            except:
              error_count += 1
              pass

        line_count += 1
        if need_select:
          total_triple_count += sum([len(q_item['claims'][predicate]) for predicate in q_item['claims']])
          selected_line_count += 1
          wf.write(line + '\n')
        if line_count % 100000 == 0:
          print('[line]:%d [selectedLine]:%d [triples]:%d [timeConsumed]:%ds' % (
            line_count, selected_line_count, total_triple_count, time.time() - time0))
          time0 = time.time()


def main():
  data_folder = r'F:\WikiData'
  wikidata_file_path = os.path.join(data_folder, 'latest-all.json')
  # selected_whole_line_data_path = os.path.join(data_folder, 'selected-latest-all.WholeLines.data')
  selected_only_relevant_data_path = os.path.join(data_folder, 'selected-latest-all.OnlyRelevant.data')

  qid_set = get_qid_set(train_file_path='../../data/trains_ansEntity_fixConnectErr.txt',
                        test_file_path='../../data/ts_ansEntity_raw.txt',
                        pkl_saving_path='../../data/qid_set.pkl')

  print('len(qid_set) = %d' % len(qid_set))
  # select_the_whole_line(qid_set, wikidata_file_path, selected_data_path=selected_whole_line_data_path)
  select_relevant_entities(qid_set, wikidata_file_path, selected_data_path=selected_only_relevant_data_path)


if __name__ == '__main__':
  main()
