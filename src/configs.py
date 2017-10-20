class TransEConfig:
  batch_size = 32         # 调这个可加速，有时对结果也有影响，能64或128最好
  entities_vocab_size = 500000
  relations_vocab_size = 3065 + 1
  embedding_size = 128    # 优先调大这个值 256或512均可

  num_sampled = 64
  init_scale = 0.5


class CNNModelConfig:
  num_sampled=64
  batch_size = 4
  base_learning_rate=0.1
  lr_decay=0.98

  max_question_length=50

  words_vocab_size = 50000
  entities_vocab_size = 500000
  relations_vocab_size = 3065 + 1
  embedding_size = 128
  output_latent_vec_size=50

  # num_sampled = 64
  init_scale = 0.5

  need_vocab_aligment = True

  margin=0.8
