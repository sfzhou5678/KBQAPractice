import tensorflow as tf

from src.top_level.QADataManager import DataDataManagerImp
from src.database.DBManager import DBManager


def func(db, data_helper, raw_data_path, new_data_path):
  wf = open(new_data_path, 'w',encoding='utf-8')
  line_count = 0
  with open(raw_data_path,encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
      datas = eval(line.strip())
      candidate_topic = datas['candidate_topic']

      candidate_relation_pids = data_helper.build_relation_set(db, candidate_topic)
      candidate_relation_pids = list(candidate_relation_pids)
      datas['candidate_relation'] = candidate_relation_pids

      wf.write(str(datas) + '\n')
      line_count += 1
      if line_count % 200 == 0:
        print(line_count)
  wf.close()


raw_train_path = r'D:\MyProjectsRepertory\PythonProject\KBQAPractice\data\SimpleQuestions\sq.train.textrazor.txt'
raw_test_path = r'D:\MyProjectsRepertory\PythonProject\KBQAPractice\data\SimpleQuestions\sq.test.textrazor.txt'

train_save_path = r'D:\MyProjectsRepertory\PythonProject\KBQAPractice\data\SimpleQuestions\sq.train.textrazor.withRelation.txt'
test_save_path = r'D:\MyProjectsRepertory\PythonProject\KBQAPractice\data\SimpleQuestions\sq.test.textrazor.withRelation.txt'

data_helper = DataDataManagerImp()

db_setting = {'host': '192.168.1.139', 'port': 3306, 'user': 'root', "psd": '1405', "db": 'knowledge_base'}
db = DBManager(host=db_setting['host'], port=db_setting['port'], user=db_setting['user'],
               psd=db_setting['psd'], db=db_setting['db'])

func(db, data_helper, raw_train_path, train_save_path)
func(db, data_helper, raw_test_path, test_save_path)
