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


def pad_realtion(relations, max_relation_size, vocab, R_UNK, R_PAD):
  """
  输入一个用P和R表示的关系列表
  :param relations:
  :param max_relation_size:
  :param vocab:
  :param R_UNK:
  :param R_PAD:
  :return:  返回Relation对应的id列表 比如P123对应8之类
  """
  relations = relations[:max_relation_size]
  new_relation_ids = [lookup_vocab(vocab, r, R_UNK) for r in relations]
  for _ in range(max_relation_size - len(new_relation_ids)):
    new_relation_ids.append(R_PAD)

  return new_relation_ids
