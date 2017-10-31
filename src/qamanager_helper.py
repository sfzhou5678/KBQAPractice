import re
import string
from abc import abstractmethod

import numpy as np
import heapq
from src.tools.common_tools import lookup_vocab
from src.preparing_data.question_entities_recognition import parse_question

punctuation = string.punctuation


class ManagerHelperInterface:
  @abstractmethod
  def recognize_topic_entity(self, question): pass

  @abstractmethod
  def question2ids(self, raw_question, word_vocab, question_max_length, padding_id):  pass

  @abstractmethod
  def candi2ids(self, candidate_data, item_vocab, relation_vocab, is_forward): pass

  @abstractmethod
  def entity2id(self, entity_qid, vocab): pass

  @abstractmethod
  def get_data_to_model(self, question, topic_entity_qid, config,
                        word_vocab, item_vocab, relation_vocab,
                        forward_candidate_data, backward_candidate_data,
                        padding_id=1):
    pass

  @abstractmethod
  def select_topk_ans(self, relations, answers, similarities, k, threshold=0.0): pass

  @abstractmethod
  def id2text(self, data_id, reverse_vocab): pass

  @abstractmethod
  def get_pred_triples(self, topic_entity_text, fw_pred_relations, fw_pred_answers, bw_pred_relations, bw_pred_answers,
                       reverse_relation_vocab, reverse_item_vocab): pass


class QAManagerHelpderImp(ManagerHelperInterface):
  def recognize_topic_entity(self, question):
    success, parsed_question = parse_question(question)

    # 4. 记录结果
    qids = []
    if success:
      if 'entities' in parsed_question:
        for entity in parsed_question['entities']:
          if 'wikidataId' in entity:
            qids.append(entity['wikidataId'])
          else:
            # todo 日志中记录这个答案没有对应的qid
            pass
      else:
        # todo 日志中记录这个问题没有找到对应的entities
        pass
      return qids
    else:
      # TODO 尚未清楚success什么情况下会是false,记录日志
      return None

  def question2ids(self, raw_question, word_vocab, question_max_length, padding_id):
    raw_question = re.sub(r'[{}]+'.format(punctuation), ' ', raw_question).strip()
    question_ids = [lookup_vocab(word_vocab, word) for word in re.split('\s+', raw_question)]
    question_ids = question_ids[:question_max_length]
    for i in range(question_max_length - len(question_ids)):
      question_ids.append(padding_id)

    return np.array(question_ids)

  def entity2id(self, entity_qid, vocab):
    id = lookup_vocab(vocab, entity_qid)
    return np.array(id)

  def get_data_to_model(self, question, topic_entity_qid, config, word_vocab, item_vocab, relation_vocab,
                        forward_candidate_data, backward_candidate_data, padding_id=1):
    # 1. question to id
    question_ids = self.question2ids(question, word_vocab,
                                     config.max_question_length, padding_id=padding_id)

    # 2. entities to id
    topic_entity_id = self.entity2id(topic_entity_qid, item_vocab)
    fw_relations, fw_answers = self.candi2ids(forward_candidate_data,
                                              item_vocab, relation_vocab,
                                              is_forward=True)
    bw_relations, bw_answers = self.candi2ids(backward_candidate_data,
                                              item_vocab, relation_vocab,
                                              is_forward=False)

    return question_ids, topic_entity_id, fw_relations, fw_answers, bw_relations, bw_answers

  def candi2ids(self, candidate_data, item_vocab, relation_vocab, is_forward):
    if is_forward:
      ids = [(lookup_vocab(relation_vocab, r), lookup_vocab(item_vocab, t))
             for (h, r, t) in candidate_data]
    else:
      ids = [(lookup_vocab(relation_vocab, r), lookup_vocab(item_vocab, h))
             for (h, r, t) in candidate_data]
    ids = [(h, r) for h, r in ids if r != 0]  # 过滤掉unk的数据

    relations = np.array([relation for relation, ans in ids])
    answers = np.array([ans for relation, ans in ids])

    return relations, answers

  def select_topk_ans(self, relations, answers, similarities, k, threshold=0.0):
    topk_index = heapq.nlargest(k, range(len(similarities)), similarities.take)
    topk_index = [index for index in topk_index if similarities[index] >= threshold]

    topk_ans = [answers[index] for index in topk_index]
    topk_relation = [relations[index] for index in topk_index]

    return topk_relation, topk_ans

  def id2text(self, data_id, reverse_vocab):
    id = lookup_vocab(reverse_vocab, data_id)
    # todo 根据QID在线获取label
    return "text" + str(id)

  def get_pred_triples(self, topic_entity_text, fw_pred_relations, fw_pred_answers, bw_pred_relations, bw_pred_answers,
                       reverse_relation_vocab, reverse_item_vocab):
    fw_pred_text_relations = [self.id2text(relation, reverse_relation_vocab)
                              for relation in fw_pred_relations]
    fw_pred_text_answers = [self.id2text(ans, reverse_item_vocab)
                            for ans in fw_pred_answers]

    bw_pred_text_relations = [self.id2text(relation, reverse_relation_vocab)
                              for relation in bw_pred_relations]
    bw_pred_text_answers = [self.id2text(ans, reverse_item_vocab)
                            for ans in bw_pred_answers]

    pred_triples = []
    for r, a in zip(fw_pred_text_relations, fw_pred_text_answers):
      triple = {"ans": a, "detail": {"head": topic_entity_text, "relation": r, "ans": a}}
      pred_triples.append(triple)
    for r, a in zip(bw_pred_text_relations, bw_pred_text_answers):
      triple = {"ans": a, "detail": {"head": a, "relation": r, "ans": topic_entity_text}}
      pred_triples.append(triple)
    pass
