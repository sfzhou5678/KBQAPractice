import os
import re
import time
import threading

from src.preparing_data.TextRazorManager import TextRazorManager

# qa_folder = r'D:\DeeplearningData\NLP-DATA\英文QA\WebQuestions'
qa_folder = r'D:\MeachineLearningData'
train_path = os.path.join(qa_folder, 'webquestions.examples.train.json')
test_path = os.path.join(qa_folder, 'webquestions.examples.test.json')

train_res_path = '../../data/webquestions.train.textrazor.full.txt'
test_res_path = '../../data/webquestions.test.textrazor.full.txt'

train_log_path = '../../log/train.log'
test_log_path = '../../log/test.log'


def load_or_create_count(log_path):
  try:
    with open(log_path) as f:
      count = int(f.readline().strip())
      return count
  except:
    return 0


def parse_question(question):
  """
  暂时只有TextRazor一种Parser
  后续可以替换成自己的
  :param question:
  :return:
  """

  def text_razor_parsing(question):
    client = TextRazorManager.get_client()
    if client is None:
      api_key = api_key_list[0]
      client = TextRazorManager.get_client_with_key(api_key)

    def do_parse(client, question):
      try:
        # TODO 还不知道Parse失败会返回什么 下次可以试一下
        response = client.analyze(question)
        success = response.ok
      except:
        # 目前只知道API次数用完会导致analyze失败
        return False, None
      parsed_question = None
      if success:
        parsed_question = response.json['response']

      return success, parsed_question

    success, parsed_question = do_parse(client, question)

    cnt = 0
    # fixme: 加了几个变态的while 有时间改掉
    while not success:
      time.sleep(1)
      success, parsed_question = do_parse(client, question)
      cnt += 1
      if cnt > 10:
        break

    error_cnt = 0
    while not success:
      error_cnt += 1
      if error_cnt > 20:
        break
      api_key_lock.acquire()
      client = TextRazorManager.get_new_client(api_key_list, client)
      api_key_lock.release()

      success, parsed_question = do_parse(client, question)
      cnt = 0
      while not success:
        time.sleep(1)
        success, parsed_question = do_parse(client, question)
        cnt += 1
        if cnt > 10:
          break

    return success, parsed_question

  success, parsed_question = text_razor_parsing(question)
  return success, parsed_question


def question_entities_recognition(question_path, res_path, log_path):
  total_handled_count = load_or_create_count(log_path)
  cur_count = 0
  # 处理question.train
  with open(question_path) as f:
    lines = f.readlines()
    with open(res_path, 'a', encoding='utf-8') as wf:
      for line in lines:
        line = line.strip()
        if line.count('utterance') > 0:
          # 0. 去掉文件开头的'['行和结尾的']'以及空行
          cur_count += 1
          if cur_count <= total_handled_count:
            continue
          qa_pair = {}
          line_dic = eval(line)
          if isinstance(line_dic, tuple):
            line_dic = line_dic[0]

          # 1. 获取question
          question = line_dic['utterance']
          qa_pair['question'] = question
          print(question)

          # 2. 获取ans
          # 2.1 原始TargetValues=(list (description Assassination) (description Firearm)
          # 所以提取出'description '以后的内容作为ans
          p = re.compile('(?<=description )[^)]+')
          answers = p.findall(line_dic['targetValue'])
          # 2.2 由于原ans中有些带""，有些不带，比如：['"Zambia"', 'Zimbabwe', '"Mozambique"', 'KwaZulu-Natal']
          # 先将所有ans统一处理成以下形式：
          # ['Zambia', 'Zimbabwe', 'Mozambique', 'KwaZulu-Natal']
          for ans, i in zip(answers, range(len(answers))):
            if ans.startswith('"'):
              answers[i] = ans[1:-1]
          qa_pair['ans'] = answers
          # print(answers)

          # 3. 调用api查询question中的entity等
          success, parsed_question = parse_question(question)
          qa_pair['parsed_question'] = parsed_question

          # 4. 记录结果
          if success:
            wf.write(str(qa_pair) + '\n')
            wf.flush()
            total_handled_count += 1
            with open(log_path, 'w') as log_wf:
              # 记录当前成功处理到哪里了
              log_wf.write(str(total_handled_count) + '\n')
          else:
            print('Error!')
            break


def sqdata_question_entities_recognition(f, wf, log_path, error_wf, db, data_helper, is_train=True):
  """
  专门处理SimpleQuestions中问题数据的函数
  :param question_path:
  :param res_path:
  :param log_path:
  :return:
  """
  global r_lock
  global w_lock
  if is_train:
    global train_total_handled_count
    total_handled_count = train_total_handled_count
  else:
    global test_total_handled_count
    total_handled_count = test_total_handled_count

  cur_count = 0
  while True:
    r_lock.acquire()
    line = f.readline().strip()
    if line == '':
      # 最后一行是一个']'
      r_lock.release()
      break

    cur_count += 1
    if cur_count <= total_handled_count:
      continue
    r_lock.release()

    head, relation, tail, question = line.split('\t')

    # 3. 调用api查询question中的entity等
    success, parsed_question = parse_question(question)

    # 4. 记录结果
    if success:
      candidate_topic = []
      if 'entities' in parsed_question:
        for entity in parsed_question['entities']:
          if 'wikidataId' in entity:
            candidate_topic.append(entity['wikidataId'])

      if head not in candidate_topic:
        candidate_topic.append(head)

      qa_pair = {}
      qa_pair['question'] = question
      qa_pair['ans'] = tail

      qa_pair['gt_topic'] = head
      qa_pair['candidate_topic'] = candidate_topic

      qa_pair['gt_relation'] = relation

      candidate_relation_pids = data_helper.build_relation_set(db, candidate_topic)
      candidate_relation_pids = list(candidate_relation_pids)
      qa_pair['candidate_relation'] = candidate_relation_pids

      w_lock.acquire()

      wf.write(str(qa_pair) + '\n')
      wf.flush()

      # 记录工作编号，最后flush
      with open(log_path, 'w') as log_wf:
        if is_train:
          train_total_handled_count += 1
          if train_total_handled_count % 100 == 0:
            print(train_total_handled_count)

          # 记录当前成功处理到哪里了
          log_wf.write(str(train_total_handled_count) + '\n')
        else:
          test_total_handled_count += 1
          if test_total_handled_count % 100 == 0:
            print(test_total_handled_count)

          # 记录当前成功处理到哪里了
          log_wf.write(str(test_total_handled_count) + '\n')
      w_lock.release()
    else:
      w_lock.acquire()
      error_wf.write(line + "\n")
      error_wf.flush()
      w_lock.release()
      continue


api_key_list = ["b061d4d2c7fade52e7ae8c3c786eb4c50eb935876aae037159de827e",
                '9ba775f9ac0a0afdbd3d6aa78be0f2e4939040540e0925ff64d63b56',
                'c4098b525a46aa1836ad8d47ca673bd1645da1e73a6cd37419cc644a',
                'fbf8ba12cb15bd99c173f43f8540c4249344b66a28041ffbe9dc73eb',
                '782c0e00f7249149ee297b6e00170de561ecb63d3947a270aeb2fff5',
                '543b5c820aedbbe051c0d1b32c19ed9406fb7fb4542036d255c3e582',
                'e2cd182b39067eb855236be5c50d940b715c309eb6e1a1fdfa4dfa54',
                '26a474702d4e403aeb15044115b7c15fe2220945b0c6c4855d295784',
                'c9f7b1be6d6fac0da7064e391af266afa93ee5c82695f7677550719d',
                '6a419870f92a8d8fee15182db8b4831a193e3b57b4fccc2f71e049e4',
                'e0a4d30f91129fd9852832bf326da617180f43b425413de67aa67867',

                "9559748fcd49b977242263cc98be3e09e60e9659186d6aa1b07ebbff",
                'c2bae67dcc9a6568d802bb8152e2d50d1f8e2bc5344ac6dc58fbf109',
                '0a43407ecf749366e3bae6ec03e092d41c4ecdab4a79f3f405bbc1b7',
                '7134e99c6b1ed923809b6ef2b92f6861c675969a94ab797e3bb4732f',
                'a5a5877cee2887aa71fd0ebd03d8d14374dc1bc9d64901406e73ae9d',
                '2469798f8a8dbc8829f88ad5f21503c4a9b7af3e0ae5f14e87e9c79f',
                'e3f85e1a00f7dd9a1b862ac1b800ad1d91a62764bc5c0a29017d42f9',
                '187dec3f979007d4d5ce7aa56333a7fe81409cb30d5d0396832bac91',
                '8bfcb980b537c8e3a3052bc84b9979c8eedb36ea8064df1ef73fcd0e',
                '390c9b7e3d691cc7b20dc7093dc790c42cc9377f59a5e15a7573a68a',
                '628190b2f03ece9e42fb86b65aab2034ee2b9f25a21f4d7f8efae556',
                '6713e829242b995b9d8dac949532c82c2b43faf660fb4267289ba6d7',

                'e9b1dfa02d6c9ad267dd61cd47057ea0c9b600dc424ae0e20ce5e5c1',
                "4bbf216410aae68a4ba65d5b4b1813daa047a410d477fa56ae95211d",
                '4d89b993286174b0841cc2b77e207aa399cf0c07d1c712a24ebd5cb3',
                'dbe7f2b359a06c909ebe03215add113848c1eaeb4c6a92eb5ce62e7e',
                'd298fbba9465fb8b49efd0dc3b67893ac16ddc4e502c83262f60ab4b',
                '637dcb57ed56ec6563afe493146db73f0000a792c6db62e4b0d9ce66',
                '35c367d9f4cda86ec8ff8d147e9a5d6837810b949e051f72b55ecc93',
                '551f731e4784133621643ce1aa83940918eb7570b23f1d099e6c4a9d',
                '820279c85351094dc1ff8afc76d31da9cfcd96408e4ed4e7a9764f7a',
                '3c11b9ececc5456ce0b713b42d2e637635563036312e615a98968ea7',
                '34849bea29ab6bb8f2d90636955cf8808833871fbdc8e7c46d0b581e',
                '6c6f944d99a5d3f2c32b7231d24321715d4df0b423559e6ff1d23d88',
                'd6db1ab4780b942aae1c58c692bbf4e59bb7ff1fe2a24cd968ed824e',
                'a5b59c6cf46c06f81c7cc8c1413124229086735f97ef763e91dc6cd0',

                ]
api_key = api_key_list[0]

r_lock = threading.RLock()
w_lock = threading.RLock()
api_key_lock = threading.RLock()
train_total_handled_count = 0
test_total_handled_count = 0


def main():
  # print('===========Train QA===========')
  # question_entities_recognition(train_path, train_res_path, train_log_path)
  #
  # print('===========Test QA===========')
  # question_entities_recognition(test_path, test_res_path, test_log_path)

  ## 处理SimpleQuestions
  # sq_data_folder = r'C:\Users\zsf\Desktop\SimpleQuestions-Wikidata\wikidata-simplequestions-master'
  sq_data_folder = r'D:\DeeplearningData\NLP-DATA\英文QA\SimpleQuestions-wikidata'

  sq_train_data_path = os.path.join(sq_data_folder, 'annotated_wd_data_train.txt')
  sq_test_data_path = os.path.join(sq_data_folder, 'annotated_wd_data_test.txt')

  sq_train_res_path = '../../data/SimpleQuestions/sq.train.textrazor.withRelation.txt'
  sq_test_res_path = '../../data/SimpleQuestions/sq.test.textrazor.withRelation.txt'

  sq_train_log_path = '../../log/SimpleQuestions/train.log'
  sq_test_log_path = '../../log/SimpleQuestions/test.log'

  sq_train_error_path = '../../log/SimpleQuestions/train_error.log'
  sq_test_error_path = '../../log/SimpleQuestions/test_error.log'

  from src.top_level.QADataManager import DataDataManagerImp
  from src.database.DBManager import DBManager

  data_helper = DataDataManagerImp()

  db_setting = {'host': '192.168.1.139', 'port': 3306, 'user': 'root', "psd": '1405', "db": 'knowledge_base'}
  db = DBManager(host=db_setting['host'], port=db_setting['port'], user=db_setting['user'],
                 psd=db_setting['psd'], db=db_setting['db'])

  global train_total_handled_count
  global test_total_handled_count

  time0 = time.time()
  thread_num = 2
  thread_list = []

  print('===========Train QA===========')
  train_total_handled_count = load_or_create_count(sq_train_log_path)
  train_f = open(sq_train_data_path, encoding='utf-8')
  train_wf = open(sq_train_res_path, 'a', encoding='utf-8')
  # train_log_wf = open(sq_train_log_path, 'w')
  train_error_wf = open(sq_train_error_path, 'a', encoding='utf-8')

  for _ in range(thread_num):
    t = threading.Thread(target=sqdata_question_entities_recognition,
                         args=(train_f, train_wf, sq_train_log_path, train_error_wf, db, data_helper))
    thread_list.append(t)
    t.start()
    time.sleep(2)

  for i in range(thread_num):
    thread = thread_list[i]
    thread.join()

  print('===========Test QA===========')
  test_total_handled_count = load_or_create_count(sq_test_log_path)
  test_f = open(sq_test_data_path, encoding='utf-8')
  test_wf = open(sq_test_res_path, 'a', encoding='utf-8')
  # test_log_wf = open(sq_test_log_path, 'w')
  test_error_wf = open(sq_test_error_path, 'a', encoding='utf-8')

  # sqdata_question_entities_recognition(test_f, test_wf, test_log_wf, test_error_wf)
  for _ in range(thread_num):
    t = threading.Thread(target=sqdata_question_entities_recognition,
                         args=(test_f, test_wf, sq_test_log_path, test_error_wf, db, data_helper, False,))
    thread_list.append(t)
    t.start()
    time.sleep(2)

  for i in range(thread_num):
    thread = thread_list[i]
    thread.join()


if __name__ == '__main__':
  main()
