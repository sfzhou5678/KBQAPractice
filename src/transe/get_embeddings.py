import tensorflow as tf
from src.transe.TransEModel import TransEModel
from src.configs import TransEConfig
from src.tools.common_tools import pickle_dump
from src.tools.db_to_model_data import get_entity_vocabulary
import os
import re
import numpy as np


def main():
  wikidata_folder = r'F:\WikiData'
  transe_data_save_path = os.path.join(wikidata_folder, 'transE.OnlyRelevant.data')
  transe_ids_save_path = os.path.join(wikidata_folder, 'transE.OnlyRelevant.ids')

  item_counter_save_path = os.path.join(wikidata_folder, 'item.counter.OnlyRelevant.cnt')
  relation_counter_save_path = os.path.join(wikidata_folder, 'relation.counter.OnlyRelevant.cnt')

  item_vocab_save_path = os.path.join(wikidata_folder, 'item.vocab.OnlyRelevant.voc')
  relation_vocab_save_path = os.path.join(wikidata_folder, 'relation.vocab.OnlyRelevant.voc')

  relation_vocab = get_entity_vocabulary(relation_counter_save_path, relation_vocab_save_path,
                                         UNK='RELATION_UNK', percent=1.0)
  item_vocab = get_entity_vocabulary(item_counter_save_path, item_vocab_save_path, UNK='ITEM_UNK', percent=0.80)

  config = TransEConfig()
  config.batch_size = 1
  config.relations_vocab_size = len(relation_vocab)
  config.entities_vocab_size = len(item_vocab)

  model_save_path = r'F:\WikiData\TransE\res-[8]-[100]-[64]\model.ckpt'
  item_embeddings_save_path = r'F:\WikiData\item.embeddings.emd'
  relation_embeddings_save_path = r'F:\WikiData\relation.embeddings.emd'

  with tf.name_scope('Train'):
    with tf.variable_scope("Model", reuse=None):
      model = TransEModel(config=config, is_training=True)

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    # tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(r'F:\WikiData\TransE\res-[8]-[100]-[64]'))

    item_embeddings = sess.run(model.entities_embeddings)
    count = 0
    np.save("F:\WikiData\item_embeddings.npy", item_embeddings)


if __name__ == '__main__':
  main()
