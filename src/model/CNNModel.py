import tensorflow as tf
from src.model.cnn_utils import *
from src.tools.common_tools import get_accuracy, nce_alignment, cos_similarity

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class CNNModel:
  def __init__(self, config, is_training=True, is_test=False):
    batch_size = config.batch_size
    words_vocab_size = config.words_vocab_size
    entities_vocab_size = config.entities_vocab_size
    relations_vocab_size = config.relations_vocab_size
    embedding_size = config.embedding_size
    max_question_length = config.max_question_length
    num_sampled = config.num_sampled
    output_latent_vec_size = config.output_latent_vec_size

    base_learning_rate = config.base_learning_rate
    lr_decay = config.lr_decay

    self.global_step = tf.contrib.framework.get_or_create_global_step()

    # 每次输入batchSize个数据
    # 每个数据包含：questionIds，一个topicEntity，一组正确答案，一组定长的负采样答案(候选答案-正确答案+填充至定长)
    # 答案拆分为：relation和tailEntity两个部分，所以应该有正确答案的R和T+错误答案的R+T共4组数据[TODO: 正确答案的长度能否变长？如果必须定长的话那就默认一次1个正确答案]

    self.question_ids = tf.placeholder(tf.int32, [batch_size, max_question_length])
    self.topic_entity_id = tf.placeholder(tf.int32, [batch_size])

    if not is_test:
      self.true_relation = tf.placeholder(tf.int32, [batch_size])
      self.true_ans = tf.placeholder(tf.int32, [batch_size])

      self.neg_relation = tf.placeholder(tf.int32, [batch_size, num_sampled])
      self.neg_ans = tf.placeholder(tf.int32, [batch_size, num_sampled])
    else:
      self.candidate_relation = tf.placeholder(tf.int32, [batch_size, None])
      self.candidate_ans = tf.placeholder(tf.int32, [batch_size, None])
    # self.ans_type = tf.placeholder(tf.int32, [None])  # TODO 可以做成one-hot
    # self.context_ids = tf.placeholder(tf.int32, [batch_size, None])  # 每个节点的context数量都不同，所以第二维设成了none

    self.is_training = is_training

    with tf.device("/cpu:0"):
      # 总共有文本word，entity以及relation三个词汇表
      words_embeddings = tf.get_variable('words_embeddings', [words_vocab_size, embedding_size], dtype=tf.float32)

      entities_embeddings = tf.get_variable('entities_embeddings', [entities_vocab_size, embedding_size],
                                            dtype=tf.float32)
      relations_embeddings = tf.get_variable('relations_embeddings', [relations_vocab_size, embedding_size],
                                             dtype=tf.float32)
      # 2D conv要求输入是4维的([batchsize,width,height,depth])
      # 而原inputs是3维的([batch_size,num_steps,embedding_size])
      embedded_question = tf.nn.embedding_lookup(words_embeddings, self.question_ids)
      embedded_question = tf.expand_dims(embedded_question, -1)

      # fixme: 对字符串形式的ans的处理方法(str、time等用一个统一的id标注，不考虑具体的值(因为只要topic和R对的话值就能确定下来了))
      embedded_entity = tf.nn.embedding_lookup(entities_embeddings, self.topic_entity_id)

      if not is_test:
        embedded_ans = tf.nn.embedding_lookup(entities_embeddings, self.true_ans)
        embedded_relation = tf.nn.embedding_lookup(relations_embeddings, self.true_relation)

        embedded_neg_ans = tf.nn.embedding_lookup(entities_embeddings, self.neg_ans)
        embedded_neg_realtion = tf.nn.embedding_lookup(relations_embeddings, self.neg_relation)

        # todo type的编码方式?
        # todo 根据ids找到一个context列表 然后reduceMean
      else:
        embedded_candidate_ans = tf.nn.embedding_lookup(entities_embeddings, self.candidate_ans)
        embedded_candidate_relation = tf.nn.embedding_lookup(entities_embeddings, self.candidate_relation)

    # TODO 这种entity可以试试直接CNN提取，也可以在TransE的基础上，试一下把ansentity_latent替换成H+R=T!
    question_ansentity_latent_vec = self.get_latent_vec(embedded_question, embedding_size, name='question_ans_entity',
                                                        reuse=not is_training)
    question_relation_latent_vec = self.get_latent_vec(embedded_question, embedding_size, name='question_relation',
                                                       reuse=not is_training)
    # question_context_latent = self.get_latent(embedded_question, config, name='question_context',
    #                                                    is_training=not is_training)
    # # fixme 如果type用one-hot的话，不知道latent函数是否要改
    # question_type_latent = self.get_latent(embedded_question, config, name='question_type',
    #                                                 is_training=is_training)

    """
    得到latent之后要做的就是让各latent和目标ans尽量接近，并且和非目标ans尽量远离
    
    ####Step1. NCE: 
    NCE的作用相当于让用于自然语言的词汇表和FB中用到的entity以及relation词汇表做一个对齐
    相当于一个预训练过程，可以单独训练，也可以一起训练(目前选择的是一起训练)
    参照NCE在word2vec中的用法，区别在于原始输入的wordEmbedding被转换成了questionLatent，其他都保持不变
    
    ####Step2. Cos similarity
    第一步NCE已经对不同词汇表的词作了对齐，那么接下来要做的就是判断QLatent和ALatents的相似度了
    关于相似度的训练目前有两种方案：
    1) 参考论文中的，分别对不同维度的特征做相似度计算，最后再求和作为总的相似度[优化目标是4+1还是4个还是1个？]
    2) 自己想的将不同纬度的latent做concat拼接到一起，然后再通过一个CNN提取一次特征，最后直接对这个提的的特征做负采样Cos相似度计算[可以简化优化目标]
    
    目前选择的是第二种方案
    
    注：基于margin loss function的cos 相似度公式：loss=max(0, m+cos(q,p)-cos(q,n))
        其中m不宜太大，一般是0.2以下的值
    
    [待补充]
    """

    need_vocab_aligment = config.need_vocab_aligment  # 是否需要做自然语言到FBEentity的对齐 如果要的话则是通过类似word2vec的nceLoss完成
    if is_training:
      if need_vocab_aligment:
        question_ansentity_nce_loss = nce_alignment(question_ansentity_latent_vec, self.true_ans,
                                                    batch_size, embedding_size,
                                                    entities_vocab_size, num_sampled,
                                                    name='question_ans_entity',
                                                    is_training=is_training)
        question_relation_nce_loss = nce_alignment(question_relation_latent_vec, self.true_relation, batch_size,
                                                   embedding_size,
                                                   relations_vocab_size, num_sampled,
                                                   name='question_relation',
                                                   is_training=is_training)
        # todo : type怎么处理？
        # question_type_latent_nce_loss = nce_alignment(question_type_latent, self.ans_type,batch_size, embedding_size,
        #                                               type_vocab_size, num_sampled,
        #                                               name='question_entity',
        #                                               is_training=is_training)

        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(
          base_learning_rate,
          global_step,
          300,
          lr_decay
        )
        self.q_entity_nce_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(question_ansentity_nce_loss)
        self.q_relation_nce_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(
          question_relation_nce_loss)

    similarity_mode = 'concat'
    if similarity_mode == 'concat':
      # fixme: 用了太多concat和reshape 不知道执行效率会受到多大的影响

      # 拼接多维度特征，分别产生qeustion,ans,neg_ans的特征向量
      question_latent_vec = tf.concat([tf.reshape(question_ansentity_latent_vec, [batch_size, 1, embedding_size]),
                                       tf.reshape(question_relation_latent_vec, [batch_size, 1, embedding_size])],
                                      axis=1)
      question_latent_vec = tf.expand_dims(question_latent_vec, axis=-1)

      if not is_test:
        ans_latent_vec = tf.concat([tf.reshape(embedded_ans, [batch_size, 1, embedding_size]),
                                    tf.reshape(embedded_relation, [batch_size, 1, embedding_size])], axis=1)
        ans_latent_vec = tf.expand_dims(ans_latent_vec, axis=-1)

        candidate_ans_latent_vec = tf.concat(
          [tf.reshape(embedded_neg_ans, [batch_size * num_sampled, 1, embedding_size]),
           tf.reshape(embedded_neg_realtion,
                      [batch_size * num_sampled, 1, embedding_size])], axis=1)
        candidate_ans_latent_vec = tf.expand_dims(candidate_ans_latent_vec, axis=-1)

        # 再通过一次CNN提取特征
        question_final_latent_vec = self.get_latent_vec(question_latent_vec,
                                                        output_latent_vec_size=output_latent_vec_size,
                                                        name='final_similarity_layer', reuse=not is_training)
        ans_final_latent_vec = self.get_latent_vec(ans_latent_vec, output_latent_vec_size=output_latent_vec_size,
                                                   name='final_similarity_layer', reuse=True)
        candidate_ans_final_latent_vec = self.get_latent_vec(candidate_ans_latent_vec,
                                                             output_latent_vec_size=output_latent_vec_size,
                                                             name='final_similarity_layer', reuse=True)

        # 之前把每个batch的numSamples个数据拼在一起，当作batchSize*NumSamples个batch，现在再reshape回来
        candidate_ans_final_latent_vec = tf.reshape(candidate_ans_final_latent_vec,
                                                    [batch_size, num_sampled, output_latent_vec_size])

        # 最后计算相似度
        cos_sim_positive = cos_similarity(question_final_latent_vec, ans_final_latent_vec)
        cos_sim_positive = tf.concat([tf.reshape(cos_sim_positive, [batch_size, 1])] * num_sampled, axis=1)

        cos_sim_negative = cos_similarity(
          tf.concat([tf.reshape(question_final_latent_vec, [batch_size, 1, output_latent_vec_size])] * num_sampled,
                    axis=1),
          candidate_ans_final_latent_vec)

        # ##注意##
        # 理论上cosP-cosN越大越好，但是如果不加限制的话模型很容易就会过拟合到1，-1的状态
        # 所以需要对模型进行限制，另cosP-cosN限制在Margin以内。Margin=0.1-0.3均能取得不错的效果，margin越大则越容易过拟合
        margin = tf.constant(config.margin, shape=[batch_size, num_sampled], dtype=tf.float32)
        zeros = tf.zeros(shape=[batch_size, num_sampled])
        cos_similarity_loss = tf.maximum(zeros, tf.subtract(margin, tf.subtract(cos_sim_positive, cos_sim_negative)))
        self.cos_similarity_loss = tf.reduce_mean(cos_similarity_loss)

        correct = tf.equal(zeros, cos_similarity_loss)
        self.accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")

        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(
          base_learning_rate,
          global_step,
          300,
          lr_decay
        )
        self.cos_sim_train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cos_similarity_loss)
      else:
        # fixme: num_sampled应该是具体的candidate数
        candidate_num = 500
        candidate_ans_latent_vec = tf.concat(
          [tf.reshape(embedded_candidate_ans, [batch_size * candidate_num, 1, embedding_size]),
           tf.reshape(embedded_candidate_relation,
                      [batch_size * candidate_num, 1, embedding_size])], axis=1)
        candidate_ans_latent_vec = tf.expand_dims(candidate_ans_latent_vec, axis=-1)

        # 再通过一次CNN提取特征
        question_final_latent_vec = self.get_latent_vec(question_latent_vec,
                                                        output_latent_vec_size=output_latent_vec_size,
                                                        name='final_similarity_layer', reuse=not is_training)
        candidate_ans_final_latent_vec = self.get_latent_vec(candidate_ans_latent_vec,
                                                             output_latent_vec_size=output_latent_vec_size,
                                                             name='final_similarity_layer', reuse=True)

        question_final_latent_vec = tf.concat(
          [tf.reshape(question_final_latent_vec, [batch_size, 1, output_latent_vec_size])] * candidate_num, axis=1)
        # 之前把每个batch的numSamples个数据拼在一起，当作batchSize*NumSamples个batch，现在再reshape回来
        candidate_ans_final_latent_vec = tf.reshape(candidate_ans_final_latent_vec,
                                                    [batch_size, candidate_num, output_latent_vec_size])

        self.cos_sim = cos_similarity(question_final_latent_vec, candidate_ans_final_latent_vec)
        pass

  def get_latent_vec(self, embedded_input, output_latent_vec_size, name, reuse):
    norm = True

    # TODO 构建网络的过程弄成for
    DEPTH1 = 64
    DEPTH2 = DEPTH1 * 2
    with tf.variable_scope(name, reuse=reuse):
      # TODO 可以吧filter的size改成embedding试试
      network = conv_2d(embedded_input, [3, 3, 1, DEPTH1], [DEPTH1], [1, 1, 1, 1], 'layer1-conv1', norm=norm,
                        is_training=self.is_training)
      network = conv_2d(network, [3, 3, DEPTH1, DEPTH1], [DEPTH1], [1, 1, 1, 1], 'layer1-conv2',
                        norm=norm, is_training=self.is_training)
      network = max_pool_2d(network, [1, 2, 2, 1], [1, 2, 2, 1], 'layer1-pool1')

      network = conv_2d(network, [3, 3, DEPTH1, DEPTH2], [DEPTH2], [1, 1, 1, 1], 'layer2-conv1',
                        norm=norm, is_training=self.is_training)
      network = conv_2d(network, [3, 3, DEPTH2, DEPTH2], [DEPTH2], [1, 1, 1, 1], 'layer2-conv2',
                        norm=norm, is_training=self.is_training)
      network = max_pool_2d(network, [1, 2, 2, 1], [1, 2, 2, 1], 'layer2-pool1')

      # 最后将CNN产生的值通过全局平均池化，再通过全连接层产生latent vector
      net = slim.avg_pool2d(network, network.get_shape()[1:3], padding='VALID', scope='AvgPool')
      # 这里不能加is_training=false，如果加了就会导致val时所有cos均为1 (原因未知，但是官方IncepResnetV2中也是恒为true的)
      net = slim.dropout(net, 0.5, is_training=True, scope='Dropout')
      net = slim.flatten(net)

      latent_vec = slim.fully_connected(net, output_latent_vec_size, activation_fn=None, scope='latent_vec')

      return latent_vec


if __name__ == '__main__':
  from src.configs import CNNModelConfig

  config = CNNModelConfig()
  model = CNNModel(config, is_training=True)

  # todo assign word_embedding
