import re
import string
import numpy as np
import heapq
from src.tools.common_tools import get_id

punctuation = string.punctuation


def question2ids(raw_question, word_vocab, question_max_length, padding_id):
  raw_question = re.sub(r'[{}]+'.format(punctuation), ' ', raw_question).strip()
  question_ids = [get_id(word_vocab, word) for word in re.split('\s+', raw_question)]
  question_ids = question_ids[:question_max_length]
  for i in range(question_max_length - len(question_ids)):
    question_ids.append(padding_id)

  return np.array(question_ids)


def candi2ids(candidate_data, item_vocab, relation_vocab, is_forward):
  if is_forward:
    ids = [(get_id(relation_vocab, r), get_id(item_vocab, t))
           for (h, r, t) in candidate_data]
  else:
    ids = [(get_id(relation_vocab, r), get_id(item_vocab, h))
           for (h, r, t) in candidate_data]
  ids = [(h, r) for h, r in ids if r != 0]  # 过滤掉unk的数据

  relations = np.array([relation for relation, ans in ids])
  answers = np.array([ans for relation, ans in ids])

  return relations, answers


def entity2id(entity_qid, vocab):
  id = get_id(vocab, entity_qid)
  return np.array(id)


def get_data_to_model(question, topic_entity_qid, config,
                      word_vocab, item_vocab, relation_vocab,
                      forward_candidate_data, backward_candidate_data,
                      padding_id=1):
  # 1. question to id
  question_ids = question2ids(question, word_vocab,
                              config.max_question_length, padding_id=padding_id)

  # 2. entities to id
  topic_entity_id = entity2id(topic_entity_qid, item_vocab)
  fw_relations, fw_answers = candi2ids(forward_candidate_data,
                                       item_vocab, relation_vocab,
                                       is_forward=True)
  bw_relations, bw_answers = candi2ids(backward_candidate_data,
                                       item_vocab, relation_vocab,
                                       is_forward=False)

  return question_ids, topic_entity_id, fw_relations, fw_answers, bw_relations, bw_answers


def select_topk_ans(relations, answers, similarities, k, threshold=0.0):
  topk_index = heapq.nlargest(k, range(len(similarities)), similarities.take)
  topk_index = [index for index in topk_index if similarities[index] >= threshold]

  topk_ans = [answers[index] for index in topk_index]
  topk_relation = [relations[index] for index in topk_index]

  return topk_relation, topk_ans
