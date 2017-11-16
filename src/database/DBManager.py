import pymysql


class DBManager:
  @DeprecationWarning
  def forward_multi_hop(self, relevant_cache, topic_entity_qid, depth, multi_hop_max_depth):
    """
    本函数用于多条查询，暂时不考虑这个情况
    :param relevant_cache:
    :param topic_entity_qid:
    :param depth:
    :param multi_hop_max_depth:
    :return:
    """
    if depth > multi_hop_max_depth:
      return
    if topic_entity_qid in relevant_cache:
      triples = relevant_cache[topic_entity_qid]
    else:
      triples = self.select_from_topic(topic_entity_qid)
      relevant_cache[topic_entity_qid] = triples
    for (h, r, t) in triples:
      self.forward_multi_hop(relevant_cache, t, depth + 1, multi_hop_max_depth)

  def __init__(self, host, port, user, psd, db):
    self.conn = pymysql.connect(host=host, port=port, user=user, passwd=psd, db=db)
    self.cur = self.conn.cursor()

  def select_from_topic(self, topic_qid, max_depth=1):
    """
    选择从topicQid出发，与topic_qid相关的的实体
    :param topic_qid:
    :return:
    """
    # TODO:  现在默认是1跳情况，并且没有做相关的处理，后续按需改进
    self.cur.execute("SELECT * FROM relation WHERE QID='%s'" % topic_qid)

    return self.cur.fetchall()

  def get_labels_by_id(self, id):
    # TODO: 等label加入之后补上真正的返回label的代码

    return 'label(%s)' % str(id)

  def select_to_topic(self, topic_qid, max_depth=1):
    self.cur.execute("SELECT * FROM relation WHERE OBJ_QID='%s'" % topic_qid)

    return self.cur.fetchall()

  def select_by_head_and_tail(self, head_qid, tail_qid):
    self.cur.execute("SELECT * FROM relation WHERE QID='%s' AND OBJ_QID='%s'" % (head_qid, tail_qid))
    # for r in self.cur.fetchall():
    #   print(r)

    return self.cur.fetchall()

  def close(self):
    self.conn.close()
