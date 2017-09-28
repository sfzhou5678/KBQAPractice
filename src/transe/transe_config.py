class TransEConfig:
  batch_size = 64
  entities_vocab_size = 500000
  relations_vocab_size = 3065 + 1
  embedding_size = 256

  num_sampled = 64
  init_scale = 0.5


class CNNModelConfig:
  batch_size = 64

  words_vocab_size = 50000
  entities_vocab_size = 500000
  relations_vocab_size = 3065 + 1
  embedding_size = 256

  num_sampled = 64
  init_scale = 0.5

  need_vocab_aligment = True

  margin=0.2
