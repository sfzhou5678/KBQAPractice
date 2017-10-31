import os
import json
from src.tools.common_tools import reverse_dict
from src.database.DBManager import DBManager
from src.tools.db_to_model_data import get_entity_vocabulary, get_word_vocabulary
from src.configs import CNNModelConfig
from src.QADataManager import DataDataManagerImp
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
    self.reverse_relation_vocab = reverse_dict(self.relation_vocab)
    self.reverse_item_vocab = reverse_dict(self.item_vocab)
    self.reverse_word_vocab = reverse_dict(self.word_vocab)

    self.model_manager = QAModelManager(self.config)
    self.data_helper = DataDataManagerImp()

  def get_answer(self, question):
    """
    1. topic entity - 返回一个识别出的qid的列表
    2. 遍历qid，分别准备输入给模型的数据(在数据库中获取正向和反向的三元组，吧所有数据都转换成Model的ID，...)
    3. 调用模型获取结果
    4. 返回结果
    :param question:
    :return:
    """
    response = {"status": '',
                "result": ''}

    # 做主题词识别
    qids = self.data_helper.recognize_topic_entity(question)
    if qids is None:
      response['status'] = 'error'
      response['result'] = "调用API失败"
      return
    elif len(qids) == 0:
      response['status'] = 'error'
      response['result'] = "没有识别出topic entity."
      return

    # 构建数据
    ans = {}
    ans["question"] = question
    ans["pred"] = []
    for topic_entity_qid in qids:
      # 首先从DB中查询正反三元组
      forward_candidate_data = list(self.db.select_from_topic(topic_entity_qid, max_depth=1))
      backward_candidate_data = list(self.db.select_to_topic(topic_entity_qid, max_depth=1))

      # 构造输入模型的数据
      question_ids, topic_entity_id, \
      fw_relations, fw_answers, \
      bw_relations, bw_answers = self.data_helper.get_data_to_model(question, topic_entity_qid, self.config,
                                                                    self.word_vocab, self.item_vocab,
                                                                    self.relation_vocab,
                                                                    forward_candidate_data, backward_candidate_data,
                                                                    padding_id=1)

      # 通过Model计算相似度
      # TODO 候选答案太多时的处理方法
      fw_similarities = self.model_manager.calc_similarity(question_ids, topic_entity_id, fw_relations, fw_answers,
                                                           is_forward=True)
      bw_similarities = self.model_manager.calc_similarity(question_ids, topic_entity_id, bw_relations, bw_answers,
                                                           is_forward=True)

      # 选择答案
      fw_pred_relations, fw_pred_answers = self.data_helper.select_topk_ans(fw_relations, fw_answers, fw_similarities,
                                                                            k=3, threshold=0.3)
      bw_pred_relations, bw_pred_answers = self.data_helper.select_topk_ans(bw_relations, bw_answers, bw_similarities,
                                                                            k=5, threshold=0.95)

      # 转化成具体的单词
      topic_entity_text = self.data_helper.id2text(topic_entity_qid, self.reverse_item_vocab)
      pred_triples = self.data_helper.get_pred_triples(topic_entity_text,
                                                       fw_pred_relations, fw_pred_answers,
                                                       bw_pred_relations, bw_pred_answers,
                                                       self.reverse_relation_vocab, self.reverse_item_vocab)

      pred = {}
      pred["topic_entity"] = topic_entity_text
      pred["pred_ans"] = pred_triples

      ans["pred"].append(pred)

    # 循环完毕之后返回结果
    response["status"] = "success"
    response["result"] = json.dumps(ans)

    return json.dumps(response)


if __name__ == '__main__':
  db_setting = {'host': '192.168.1.139', 'port': 3306, 'user': 'root', "psd": '1405', "db": 'kbqa'}
  qa_sys = QASysManager(db_setting)

  print(qa_sys.get_answer("what is the oregon ducks 2012 football schedule?"))
