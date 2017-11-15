import os
import time
import json
import pickle
from src.tools.common_tools import pickle_load, pickle_dump

"""
根据QA的QID，从完整200多G的wikiJson数据中筛选数据
筛选条件：
1. 若某行数据自己的QID或其claims中涉及的QID在QA-QID中出现过，那么就将其保留
2. (暂时不考虑这一条)按0%的比例，额外随机选取数据
"""


def get_qid_set(webQ_file_paths,
                simpleQ_file_paths,
                entity_set_pkl_save_path=None,
                relation_set_pkl_sav_path=None):
  """
  ### 一些相关统计：
  1. 训练集+测试集的[topicEntities]总计有6041个，2491种(有3550个重复)
  2. 训练集+测试集的[ansEntities]总计有52291个，23124种(有29167个重复)
  3. 训练集+测试集的[topicEntities+ansEntities]总计有58332个，24945种(33387个重复)
  """
  if entity_set_pkl_save_path is not None and relation_set_pkl_sav_path is not None:
    try:
      entity_set = pickle_load(entity_set_pkl_save_path)
      relation_set = pickle_load(relation_set_pkl_sav_path)

      return entity_set, relation_set
    except:
      pass

  entity_set = set()
  relation_set = set()

  reduplicated_qid_count = 0
  for path in webQ_file_paths:
    with open(path) as f:
      lines = f.readlines()
      for line in lines:
        data = eval(line.strip())

        if 'entities' in data['parsed_question']:
          entities = data['parsed_question']['entities']
          for entity in entities:
            if 'wikidataId' in entity:
              topic_entity_qid = entity['wikidataId']
              if topic_entity_qid in entity_set:
                reduplicated_qid_count += 1
              entity_set.add(topic_entity_qid)

        for ans in data['ans']:
          for ans_entity in ans['entities']:
            ans_qid = ans_entity['Qid']
            if ans_qid in entity_set:
              reduplicated_qid_count += 1
            entity_set.add(ans_qid)

  for file_path in simpleQ_file_paths:
    with open(file_path, encoding='utf-8') as f:
      lines = f.readlines()
      for line in lines:
        line = line.strip()
        head, relation, tail, _ = line.split('\t')

        entity_set.add(head)
        entity_set.add(tail)
        relation = 'P' + relation[1:]
        relation_set.add(relation)

  if entity_set_pkl_save_path is not None:
    try:
      f_write = open(entity_set_pkl_save_path, 'wb')
      pickle.dump(entity_set, f_write, True)
    except:
      pass
  if relation_set_pkl_sav_path is not None:
    try:
      f_write = open(relation_set_pkl_sav_path, 'wb')
      pickle.dump(relation_set, f_write, True)
    except:
      pass

  return entity_set, relation_set


def select_relevant_entities(entity_set, relation_set, wikidata_file_path, selected_data_path):
  """
  筛选规则：
  1. 对于每一行数据，若这个entity本身位于qidSet中，那么该entity的信息会得到保留(但是不保证存在三元组)
  2. 对于所有与该entity有关联的实体，只将QID位于entity_set中的关系对记录下来，其余的数据抛弃





  # 以下为老规则
  # 1. 对于每一行数据，若这个entity本身位于qidSet中，就将这一整行数据保存下来
  # 2. 如果某个P位于PID_SET中，那么就将这个P一下的所有实体都保留下来
  # 3. 对于所有与该entity有关联的实体，只将QID位于qidSet中的关系对记录下来，其余的数据抛弃


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

        if QID in entity_set:
          need_select = True

        claims = {}
        for predicate in q_item['claims']:
          relevant_entities = q_item['claims'][predicate]
          relations = []
          for revelant_entity in relevant_entities:
            try:
              main_snak = revelant_entity['mainsnak']
              if main_snak['datavalue']['type'] == 'wikibase-entityid':
                revelant_qid = main_snak['datavalue']['value']['id']
                if revelant_qid in entity_set:
                  total_triple_count += 1
                  relations.append(revelant_entity)
                  need_select = True
            except:
              error_count += 1
              pass
          if len(relations) > 0:
            claims[predicate] = relations
        q_item['claims'] = claims  # 在新模式下，这个claim可能为空
        data_to_write = json.dumps(q_item)

        line_count += 1
        if need_select:
          selected_line_count += 1
          wf.write(data_to_write + '\n')
          pass
        if line_count % 100000 == 0:
          wf.flush()
          print('[line]:%d [selectedLine]:%d [triples]:%d [timeConsumed]:%ds' % (
            line_count, selected_line_count, total_triple_count, time.time() - time0))
          time0 = time.time()
      print('[line]:%d [selectedLine]:%d [triples]:%d [timeConsumed]:%ds' % (
        line_count, selected_line_count, total_triple_count, time.time() - time0))


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
  selected_only_relevant_data_path = os.path.join(data_folder, 'SimpleQeustionBased',
                                                  'selected-latest-all.OnlyRelevant.data')

  webqeustion_file_paths = ['../../data/trains_ansEntity_fixConnectErr.txt',
                            '../../data/ts_ansEntity_raw.txt']

  sq_data_folder = r'D:\DeeplearningData\NLP-DATA\英文QA\SimpleQuestions-wikidata'
  simplequestion_file_paths = [os.path.join(sq_data_folder, 'annotated_wd_data_train.txt'),
                               os.path.join(sq_data_folder, 'annotated_wd_data_valid.txt'),
                               os.path.join(sq_data_folder, 'annotated_wd_data_test.txt')]

  entity_set, relation_set = get_qid_set(webqeustion_file_paths, simplequestion_file_paths,
                                         entity_set_pkl_save_path='../../data/entity_set.pkl',
                                         relation_set_pkl_sav_path='../../data/relation_set.pkl')

  print('len(entity_set) = %d' % len(entity_set))
  # select_the_whole_line(entity_set, wikidata_file_path, selected_data_path=selected_whole_line_data_path)
  select_relevant_entities(entity_set, relation_set, wikidata_file_path,
                           selected_data_path=selected_only_relevant_data_path)


if __name__ == '__main__':
  main()
