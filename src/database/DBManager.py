import pymysql


class DBManager:
  def __init__(self, host, port, user, psd, db):
    self.conn = pymysql.connect(host=host, port=port, user=user, passwd=psd, db=db)
    self.cur = self.conn.cursor()

  def select_by_topic(self, topic_qid):
    self.cur.execute("SELECT * FROM relation WHERE QID='%s'" % topic_qid)
    # for r in self.cur.fetchall():
    #   print(r)
    return self.cur.fetchall()

  def select_by_head_and_tail(self, head_qid, tail_qid):
    self.cur.execute("SELECT * FROM relation WHERE QID='%s' AND OBJ_QID='%s'" % (head_qid, tail_qid))
    # for r in self.cur.fetchall():
    #   print(r)

    return self.cur.fetchall()
  def close(self):
    self.conn.close()
