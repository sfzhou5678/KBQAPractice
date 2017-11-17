import tensorflow as tf
from src.configs import RNNModelConfig
from src.model.RNNModel import RNNModel
import numpy as np
from src.tools.char_rnn_tools import pad_item, pad_realtion


def pred():
  config = RNNModelConfig()
  model = RNNModel(config=config, is_training=True, is_testing=False)

  with tf.Session() as sess:
    tf.global_variables_initializer().run()

    question = np.reshape(range(config.batch_size * config.max_question_length),
                          [config.batch_size, config.max_question_length])
    candidate_items = np.random.randint(config.char_vocab_size,
                                        size=[config.batch_size, config.max_candidate_item_size,
                                              config.max_item_label_length])
    candidate_relations = np.random.randint(config.relations_vocab_size,
                                            size=[config.batch_size, config.max_candidate_relation_size])

    gt_items = np.random.randint(config.max_candidate_item_size,size=[config.batch_size])
    gt_relations = np.random.randint(config.max_candidate_relation_size,size=[config.batch_size])

    for i in range(20000):
      _, _, item_loss, relation_loss, item_acc, relation_acc = sess.run(
        [model.item_train_opt, model.relation_train_opt,
         model.item_loss, model.relation_loss, model.item_acc, model.relation_acc],
        {model.question: question,
         model.candidate_items: candidate_items,
         model.candidate_relations: candidate_relations,
         model.gt_items: gt_items, model.gt_relations: gt_relations})

      if i % 100 == 0:
        print(i)
        print(item_loss, item_acc)
        print(relation_loss, relation_acc)


if __name__ == '__main__':
  pred()
