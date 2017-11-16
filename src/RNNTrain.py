import tensorflow as tf
from src.configs import RNNModelConfig
from src.model.RNNModel import RNNModel
from src.tools.common_tools import lookup_vocab


def pad_item(candidate, max_candidate_size, max_label_length,
             vocab, UNK, START, END, PAD):
  """
  按照2016EMNLP的写法制作数据
  :param candidate: 一个label列表
  :return:
  """
  candidate = candidate[:max_candidate_size]
  new_item_list = []
  for item in candidate:
    n_item = [START]
    for char in list(item[:max_label_length - 2]):
      n_item.append(lookup_vocab(vocab, char, unk_id=UNK))
    n_item.append(END)
    for _ in range((max_label_length - len(n_item))):
      n_item.append(PAD)
    new_item_list.append(n_item)

  for _ in range(max_candidate_size - len(new_item_list)):
    n_item = [START, END]
    for _ in range(max_label_length - 2):
      n_item.append(PAD)
    new_item_list.append(n_item)

  # for item in candidate_item:
  #   item = P_START + item[:max_item_label_length - 2]
  #   item += P_END
  #   item += P_PAD * (max_item_label_length - 1 - len(item))
  #   new_item_list.append(list(item))
  # for _ in range(max_candidate_item_size - len(new_item_list)):
  #   new_item_list.append(P_START + P_END+ P_PAD * (max_item_label_length - 2) )

  return new_item_list

def pad_realtion(relations,max_relation_size,vocab,R_UNK,R_PAD):
  """
  输入一个用P和R表示的关系列表
  :param relations:
  :param max_relation_size:
  :param vocab:
  :param R_UNK:
  :param R_PAD:
  :return:  返回Relation对应的id列表 比如P123对应8之类
  """
  relations=relations[:max_relation_size]
  new_relation_ids=[lookup_vocab(vocab,r,R_UNK) for r in relations]
  for _ in range(max_relation_size-len(new_relation_ids)):
    new_relation_ids.append(R_PAD)

  return new_relation_ids



def pred():
  config = RNNModelConfig()

  with tf.name_scope('Pred'):
    with tf.variable_scope("Model", reuse=None):
      pred_model = RNNModel(config, is_training=True)

  question = r'what'
  candidate_item = ['hello']
  candidate_relation = ['father of']

  vocab = {'w': 1, 'h': 2}

  padded_question = pad_item([question], 1, config.max_question_length, vocab,
                             UNK=0, START=1, END=2, PAD=3)[0]
  padded_candidate_item = pad_item(candidate_item, config.max_candidate_item_size, config.max_item_label_length,
                                   vocab, UNK=0, START=1, END=2, PAD=3)
  padded_candidate_relation = pad_item(candidate_relation, config.max_candidate_relation_size,
                                       config.max_relation_label_length, vocab,
                                       UNK=0, START=1, END=2, PAD=3)
  print(padded_candidate_relation)


if __name__ == '__main__':
  pred()
