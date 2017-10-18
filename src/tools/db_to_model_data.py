import os
import json
from src.database.DBManager import DBManager


def multi_hop(db, relevant_cache, hop_dict, topic_entity_qid, depth, multi_hop_max_depth):
  if depth > multi_hop_max_depth:
    return
  if topic_entity_qid in relevant_cache:
    triples = relevant_cache[topic_entity_qid]
  else:
    triples = db.select_by_topic(topic_entity_qid)
    relevant_cache[topic_entity_qid] = triples
  for (h, r, t) in triples:
    hop_dict[depth].add(t)
    multi_hop(db, relevant_cache, hop_dict, t, depth + 1, multi_hop_max_depth)


def main():
  db = DBManager(host='192.168.1.139', port=3306, user='root', psd='1405', db='kbqa')

  # topic_qid = 'Q12312'
  # tail_qid = 'q6683'
  #
  # db.select_by_topic(topic_qid)
  # print('=' * 80)
  # db.select_by_head_and_tail(topic_qid, tail_qid)
  train_file_path = '../../data/trains_ansEntity_fixConnectErr.txt'

  relations_counter = {}
  total_qa_pairs = 0
  has_ans_count = 0
  line_count = 0
  multi_hop_max_depth = 1

  relevant_cache = {}
  entities_multi_hop_cache = {}

  with open(train_file_path) as f:
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
            # FIXME: 目前的处理方法是直接采用第一个能在DB中找到答案的qid作为答案的Qid(即缺乏真正的消歧步骤)

            # print('=' * 80)
            # print(data['question'], entity['entityId'],[ans['name'] for ans in data['ans']])
            # func1 标准1跳
            #
            for ans in data['ans']:
              total_qa_pairs += 1
              for ans_entity in ans['entities']:
                ans_qid = ans_entity['Qid']
                relevant_triples = db.select_by_head_and_tail(topic_entity_qid, ans_qid)

                if len(relevant_triples) > 0:
                  # print('√',ans_entity['label'],',',ans_entity['description'])
                  has_ans_count += 1
                  # break
                # else:
                #   reversed_relevant_triples = db.select_by_head_and_tail(ans_qid, topic_entity_qid, )
                #   if len(reversed_relevant_triples) > 0:
                #     # print('R√',ans_entity['label'],',',ans_entity['description'])
                #     has_ans_count += 1
                #     break
                #   else:
                #     print('×',ans_entity['label'],',',ans_entity['description'])
                #     pass

                    # func2 多跳查询
                    #
                    # if topic_entity_qid in entities_multi_hop_cache:
                    #   hop_dict = entities_multi_hop_cache[topic_entity_qid]
                    # else:
                    #   hop_dict = {}  # hop_dict[n]表示n跳所能到达的答案
                    #   for i in range(multi_hop_max_depth):
                    #     hop_dict[i + 1] = set()
                    #   multi_hop(db, relevant_cache,hop_dict, topic_entity_qid, depth=1, multi_hop_max_depth=multi_hop_max_depth)
                    #   entities_multi_hop_cache[topic_entity_qid] = hop_dict
                    #
                    # for ans in data['ans']:
                    #   total_qa_pairs += 1
                    #   for ans_entity in ans['entities']:
                    #     ans_qid = ans_entity['Qid']
                    #     for i in range(multi_hop_max_depth):
                    #       if ans_qid in hop_dict[i + 1]:
                    #         has_ans_count += 1
                    #         break
          # print(relations_counter)
        if line_count % 100 == 0:
          print(total_qa_pairs)
          print(has_ans_count)
          print('=' * 80)
  print(total_qa_pairs)
  print(has_ans_count)
  # print(relations_counter)

  db.close()
  pass


if __name__ == '__main__':
  main()
