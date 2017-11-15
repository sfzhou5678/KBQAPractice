import os
import numpy
import time
import json


def get_sq_dict(file_paths):
  """
  :param file_paths: 一个文件名列表，比如train+val+test
  :return: 以头为key做第一层dict，p为key做第二层dict,里面的值做list
  """
  sq_dict = {}
  rsq_dict = {}
  total_lines = 0
  P_count = 0
  R_count = 0
  for file_path in file_paths:
    with open(file_path, encoding='utf-8') as f:
      lines = f.readlines()
      total_lines += len(lines)
      for line in lines:
        line = line.strip()
        head, relation, tail, _ = line.split('\t')
        if relation.startswith('P'):
          P_count += 1
          if head not in sq_dict:
            sq_dict[head] = {}
          if relation not in sq_dict[head]:
            sq_dict[head][relation] = {}
          sq_dict[head][relation][tail] = 0
        if relation.startswith('R'):
          R_count += 1
          relation = 'P' + relation[1:]
          t = tail
          tail = head
          head = t
          if head not in rsq_dict:
            rsq_dict[head] = {}
          if relation not in rsq_dict[head]:
            rsq_dict[head][relation] = {}
          rsq_dict[head][relation][tail] = 0

  print(total_lines)
  total_item_count = 0
  for k1 in sq_dict:
    for k2 in sq_dict[k1]:
      for k3 in sq_dict[k1][k2]:
        total_item_count += 1

  rtotal_item_count = 0
  for k1 in rsq_dict:
    for k2 in rsq_dict[k1]:
      for k3 in rsq_dict[k1][k2]:
        rtotal_item_count += 1
  print(total_item_count, rtotal_item_count)
  print(P_count, R_count)
  print(len(sq_dict), len(rsq_dict))
  return sq_dict, rsq_dict


def check_wikidata(wikidata_file_path, sq_dict, rsq_dict):
  line_count = 0
  hit_count = 0
  rhit_count = 0
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

      q_item = json.loads(line)
      QID = q_item['id']

      if QID in sq_dict:
        for predicate in q_item['claims']:
          if predicate in sq_dict[QID]:
            relevant_entities = q_item['claims'][predicate]
            for revelant_entity in relevant_entities:
              try:
                main_snak = revelant_entity['mainsnak']
                # if main_snak['datavalue']['type'] == 'wikibase-entityid':
                revelant_qid = main_snak['datavalue']['value']['id']
                if revelant_qid in sq_dict[QID][predicate]:
                  sq_dict[QID][predicate][revelant_qid] = 1
                  hit_count += 1
              except:
                pass

      if QID in rsq_dict:
        for predicate in q_item['claims']:
          if predicate in rsq_dict[QID]:
            relevant_entities = q_item['claims'][predicate]
            for revelant_entity in relevant_entities:
              try:
                main_snak = revelant_entity['mainsnak']
                # if main_snak['datavalue']['type'] == 'wikibase-entityid':
                revelant_qid = main_snak['datavalue']['value']['id']
                if revelant_qid in rsq_dict[QID][predicate]:
                  rsq_dict[QID][predicate][revelant_qid] = 1
                  rhit_count += 1
              except:
                pass

      line_count += 1
      if line_count % 100000 == 0:
        gt_hit_count = 0
        total_item_count = 0
        for k1 in sq_dict:
          for k2 in sq_dict[k1]:
            for k3 in sq_dict[k1][k2]:
              gt_hit_count += sq_dict[k1][k2][k3]
              total_item_count += 1

        # print('[line]:%d [selectedLine]:%d [triples]:%d [timeConsumed]:%ds' % (
        #   line_count, selected_line_count, total_triple_count, time.time() - time0))
        print('[line]:%d, [hit]:%d, [rhit]:%d [gt_hit]:%d [total]:%d' % (line_count, hit_count, rhit_count,
                                                                         gt_hit_count, total_item_count))
        time0 = time.time()


def check_if_wikidata_in_sq():
  """
  用于观察SQ中的关系对是否真的存在于完整的wikidata中(因为SQ的关系在根据WQ筛选所得的子库中几乎不存在)
  (1) 为新问答数据以头为key做dict，p做dict,里面的值做list(统计p的占比，以下只考虑p)
  (2) 遍历100万行wiki，看问答对在知识库中是否存在
  # 如果前2步成功的话：
  (3) 制作一个wq+sq+随机采样的实体集合，根据此集合重新筛选数据并且更新数据库
  :return:
  """

  sq_data_folder = r'D:\DeeplearningData\NLP-DATA\英文QA\SimpleQuestions-wikidata'
  sq_file_paths = [os.path.join(sq_data_folder, 'annotated_wd_data_train.txt'),
                   os.path.join(sq_data_folder, 'annotated_wd_data_valid.txt'),
                   os.path.join(sq_data_folder, 'annotated_wd_data_test.txt')]
  wikidata_file_path = r'F:\WikiData\latest-all.json'

  sq_dict, rsq_dict = get_sq_dict(sq_file_paths)

  check_wikidata(wikidata_file_path, sq_dict, rsq_dict)


def main():
  check_if_wikidata_in_sq()


if __name__ == '__main__':
  main()
