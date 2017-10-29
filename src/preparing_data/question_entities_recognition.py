import os
import re

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
    while not success:
      if len(api_key_list) > 1:
        api_key_list.remove(api_key_list[0])
        old_api_key = TextRazorManager.api_key
        api_key = api_key_list[0]
        print('[Change %s -> %s]' % (old_api_key, api_key))
        client = TextRazorManager.get_client_with_key(api_key)
        success, parsed_question = do_parse(client, question)
      else:
        # 最坏情况没有可用key了，就保存处理结果,停止运行
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


api_key_list = ["b061d4d2c7fade52e7ae8c3c786eb4c50eb935876aae037159de827e",
                '9ba775f9ac0a0afdbd3d6aa78be0f2e4939040540e0925ff64d63b56',
                'c4098b525a46aa1836ad8d47ca673bd1645da1e73a6cd37419cc644a',
                'fbf8ba12cb15bd99c173f43f8540c4249344b66a28041ffbe9dc73eb',
                '782c0e00f7249149ee297b6e00170de561ecb63d3947a270aeb2fff5',
                '543b5c820aedbbe051c0d1b32c19ed9406fb7fb4542036d255c3e582',
                'e2cd182b39067eb855236be5c50d940b715c309eb6e1a1fdfa4dfa54',
                ]
api_key = api_key_list[0]

if __name__ == '__main__':
  print('===========Train QA===========')
  question_entities_recognition(train_path, train_res_path, train_log_path)

  print('===========Test QA===========')
  question_entities_recognition(test_path, test_res_path, test_log_path)
