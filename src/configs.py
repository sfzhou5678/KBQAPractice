class TransEConfig:
  batch_size = 8  # 调这个可加速，有时对结果也有影响，能64或128最好
  entities_vocab_size = 5000000
  relations_vocab_size = 3065 + 1
  embedding_size = 100  # 优先调大这个值 256或512均可

  num_sampled = 64
  init_scale = 0.5
  base_learning_rate = 0.1


class CNNModelConfig:
  num_sampled = 64
  batch_size = 4
  base_learning_rate = 0.1
  lr_decay = 0.98

  max_question_length = 20

  words_vocab_size = 50000
  entities_vocab_size = 500000
  relations_vocab_size = 3065 + 1
  # 重要：三个embeddingSize最好一致
  word_embedding_size = 100
  entity_embedding_size = 100
  output_latent_vec_size = 100  # 这个稍微小一点比较好？ 可能是小的话不容易过拟合

  init_scale = 0.5

  need_vocab_aligment = True

  margin = 0.8


class RNNModelConfig:
  batch_size = 4
  num_sampled = 64

  base_learning_rate = 0.1
  lr_decay = 0.98

  max_question_length = 20

  max_candidate_item_size = 16  # size表示候选答案集合的大小
  max_item_label_length = 24    # length表示label的最大长度

  max_candidate_relation_size = 128
  max_relation_label_length = 24

  # 重要：三个embeddingSize最好一致
  word_embedding_size = 100
  entity_embedding_size = 100
  output_latent_vec_size = 100  # 这个稍微小一点比较好？ 可能是小的话不容易过拟合
