import os
from src.preparing_data.question_entities_recognition import parse_question
from src.database.DBManager import DBManager
from src.tools.db_to_model_data import get_entity_vocabulary, get_word_vocabulary
from src.configs import CNNModelConfig
from src.qamanager_helper import *
from src.QAModelManager import QAModelManager


class QASysManager(object):
  """
  服务器调用此类获取答案
  """
  wikidata_folder = r'F:\WikiData'

  selected_wikidata_file_path = os.path.join(wikidata_folder, 'selected-latest-all.OnlyRelevant.data')

  transe_data_save_path = os.path.join(wikidata_folder, 'transE.OnlyRelevant.data')
  item_counter_save_path = os.path.join(wikidata_folder, 'item.counter.OnlyRelevant.cnt')
  relation_counter_save_path = os.path.join(wikidata_folder, 'relation.counter.OnlyRelevant.cnt')

  item_vocab_save_path = os.path.join(wikidata_folder, 'item.vocab.OnlyRelevant.voc')
  relation_vocab_save_path = os.path.join(wikidata_folder, 'relation.vocab.OnlyRelevant.voc')

  pretrained_wordvec_save_path = os.path.join(wikidata_folder, 'glove.6B.100d.txt')
  word_vocab_save_path = os.path.join(wikidata_folder, 'word.vocab.glove.100.voc')
  word_embedding_save_path = os.path.join(wikidata_folder, 'word.embedding.glove.100.emd')

  item_embeddings_path = os.path.join(wikidata_folder, 'item_embeddings.npy')
  relation_embeddings_path = os.path.join(wikidata_folder, 'relation.embeddings.emd')

  def __init__(self, db_setting):
    """
    1. DB初始化
    """
    self.db = DBManager(host=db_setting['host'], port=db_setting['port'], user=db_setting['user'],
                        psd=db_setting['psd'], db=db_setting['db'])

    self.config = CNNModelConfig()

    self.relation_vocab = get_entity_vocabulary(self.relation_counter_save_path, self.relation_vocab_save_path,
                                                UNK='RELATION_UNK', percent=1.0)
    self.item_vocab = get_entity_vocabulary(self.item_counter_save_path, self.item_vocab_save_path,
                                            UNK='ITEM_UNK', percent=0.80)
    self.word_vocab, _ = get_word_vocabulary(self.pretrained_wordvec_save_path,
                                             self.word_vocab_save_path,
                                             self.word_embedding_save_path,
                                             UNK='WORD_UNK', PAD='PAD')

    self.model_manager = QAModelManager(self.config)

  def get_answer(self, question):
    """
    1. topic entity - 返回一个识别出的qid的列表
    2. 遍历qid，分别准备输入给模型的数据(在数据库中获取正向和反向的三元组，吧所有数据都转换成Model的ID，...)
    3. 调用模型获取结果
    4. 返回结果
    :param question:
    :return:
    """
    cur_log = {}

    # 做主题词识别
    qids = self.__recognize_topic_entity(question)
    if qids is None:
      cur_log['error'] = "调用API失败"
      return
    elif len(qids) == 0:
      cur_log['error'] = "没有识别出topic entity."
      return

    # 构建数据
    for topic_entity_qid in qids:
      # 首先从DB中查询正反三元组
      forward_candidate_data = list(self.db.select_from_topic(topic_entity_qid, max_depth=1))
      backward_candidate_data = list(self.db.select_to_topic(topic_entity_qid, max_depth=1))

      # 构造输入模型的数据
      question_ids, topic_entity_id, \
      fw_relations, fw_answers, bw_relations, bw_answers = get_data_to_model(question, topic_entity_qid, self.config,
                                                                             self.word_vocab, self.item_vocab,
                                                                             self.relation_vocab,
                                                                             forward_candidate_data,
                                                                             backward_candidate_data, padding_id=1)

      # 通过Model计算相似度
      # TODO 候选答案太多时的处理方法
      fw_similarities = self.model_manager.calc_similarity(question_ids, topic_entity_id,
                                                           fw_relations, fw_answers,
                                                           is_forward=True)
      bw_similarities = self.model_manager.calc_similarity(question_ids, topic_entity_id,
                                                           bw_relations, bw_answers,
                                                           is_forward=True)

      # 选择答案
      fw_pred_relations, fw_pred_answers = select_topk_ans(fw_relations, fw_answers, fw_similarities,
                                                           k=3, threshold=0.3)
      bw_pred_relations, bw_pred_answers = select_topk_ans(bw_relations, bw_answers, bw_similarities,
                                                           k=5, threshold=0.95)

      # TODO 转化成具体的单词
      # TODO 记录到一个数组中
      print()
    print()
    # TODO 循环完毕之后返回结果

  def __recognize_topic_entity(self, question):
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


if __name__ == '__main__':
  db_setting = {'host': '192.168.1.139', 'port': 3306, 'user': 'root', "psd": '1405', "db": 'kbqa'}
  qa_model = QASysManager(db_setting)

  qa_model.get_answer("what is the oregon ducks 2012 football schedule?")
