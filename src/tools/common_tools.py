import tensorflow as tf
import pickle


def cos_similarity(x1, x2):
  x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=-1))
  x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=-1))
  # 内积
  x1_x2 = tf.reduce_sum(tf.multiply(x1, x2), axis=-1)
  cos_sim = tf.divide(x1_x2, tf.multiply(x1_norm, x2_norm))

  return cos_sim


def get_accuracy(logits, target, k_list):
  acc = []
  for k in k_list:
    topk_correction_predcition = tf.nn.in_top_k(predictions=logits, targets=target, k=k)
    softmax_topk_accuracy = tf.reduce_mean(tf.cast(topk_correction_predcition, tf.float32))
    acc.append(softmax_topk_accuracy)
  return acc


def nce_alignment(embed, target_id, batch_size, embedding_size, vocab_size, num_sampled, name, is_training):
  with tf.variable_scope(name, reuse=not is_training):
    nce_weights = tf.get_variable('nce_weights', [vocab_size, embedding_size], dtype=tf.float32, )
    nce_biases = tf.get_variable('nce_biases', [vocab_size], dtype=tf.float32, )
    nce_loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=tf.reshape(target_id, [batch_size, 1]),
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocab_size))
    return nce_loss


@DeprecationWarning
def softmax_model(embed, target_id, embedding_size, vocab_size, name, is_training):
  """
  暂时还用不上
  """
  with tf.variable_scope(name, reuse=not is_training):
    softmax_weights = tf.get_variable('softmax_weights', [embedding_size, vocab_size], dtype=tf.float32)
    softmax_biases = tf.get_variable('softmax_biases', [vocab_size], dtype=tf.float32)
    softmax_logits = tf.matmul(embed, softmax_weights) + softmax_biases
    softmax_pred = tf.argmax(tf.nn.softmax(softmax_logits), axis=-1)

    softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=target_id, logits=softmax_logits, name='softmax_loss')
    softmax_loss = tf.reduce_mean(softmax_loss)

    return softmax_logits, softmax_loss, softmax_pred


def pickle_dump(data, file_path):
  f_write = open(file_path, 'wb')
  pickle.dump(data, f_write, True)


def pickle_load(file_path):
  f_read = open(file_path, 'rb')
  data = pickle.load(f_read)

  return data


def reverse_dict(cur_dict):
  new_dict = {}
  for key, value in cur_dict.items():
    new_dict[value] = key
  return new_dict


def lookup_vocab(vocab, word, unk_id=0):
  if word in vocab:
    return vocab[word]
  else:
    return unk_id

def _get_single_cell(hidden_size, keep_prob, is_training):
  cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
  if is_training and keep_prob < 1:
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
  return cell

def build_decoder_cell_with_att(encoder_outputs, encoder_final_state,
                                batch_size, max_length,
                                rnn_layers, hidden_size, keep_prob, is_training):

  # 如果只有decoder用了MultiRNNCell而encoder用的是BasicCell那么就会报错(不一致就会报错)
  decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
    [_get_single_cell(hidden_size, keep_prob, is_training) for _ in range(rnn_layers)])

  ## Create an attention mechanism
  # TODO 这里的memory_sequence_length表示source_sequence_length(输入，不是输出targets)中的非PAD的长度
  memory = encoder_outputs
  attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    hidden_size, memory,
    # memory_sequence_length=seq_lengths,
    memory_sequence_length=tf.constant(max_length, shape=[batch_size],
                                       dtype=tf.int32))

  # alignment_history = tf.cond(is_training, lambda: False, lambda: True)
  decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    decoder_cell, attention_mechanism,
    attention_layer_size=hidden_size,
    # alignment_history=False,  # test时为true
    name="attention")

  attention_states = decoder_cell.zero_state(batch_size, tf.float32).clone(
    cell_state=encoder_final_state)
  decoder_init_state = attention_states

  return decoder_cell, decoder_init_state


def build_decoder_cell_wo_att(encoder_final_state,batch_size, rnn_layers, hidden_size, keep_prob, is_training):

  # 如果只有decoder用了MultiRNNCell而encoder用的是BasicCell那么就会报错(不一致就会报错)
  decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
    [_get_single_cell(hidden_size, keep_prob, is_training) for _ in range(rnn_layers)])
  decoder_init_state=encoder_final_state

  return decoder_cell, decoder_init_state
