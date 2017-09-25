import os

data_folder = '../data'
full_train_response_path = os.path.join(data_folder, 'webquestions.train.textrazor.full.txt')
full_test_response_path = os.path.join(data_folder, 'webquestions.test.textrazor.full.txt')

train_qa_entities_pair_path = os.path.join(data_folder, 'webquestions.train.textrazor.entities.txt')
train_entity_recognition_failed_list_path = os.path.join(data_folder, 'webquestions.train.textrazor.ERFailed.txt')
test_qa_entities_pair_path = os.path.join(data_folder, 'webquestions.test.textrazor.entities.txt')
test_entity_recognition_failed_list_path = os.path.join(data_folder, 'webquestions.test.textrazor.ERFailed.txt')

# TODO: ans的entity识别！ 然后把识别结果一起写进去
"""
注意事项：
1. 有些ans可能找不到对应的entity，比如1998-01-01T00:00:00.000-05:00
统计一下这种ans的数量，如果数量很少的话就把他丢掉
2. 某些问题会的ans会是一个list，那么应该记录下所有ans对应的信息
3. 某些ans在wikidata中查询后可能会对应多个entity，如'Orange County' """  # TODO: [这时还不知道该怎么办]"""
"""
主要步骤：
1. 把ans放在wikidata的
https://www.wikidata.org/w/api.php?action=wbsearchentities&search=pudge%20rodriguez&language=en&limit=20&format=json
API下搜索entityId
PS: 所有空格都替换成%20
2. 然后
再根据entityId获取具体的信息
https://www.wikidata.org/w/api.php?action=wbgetentities&ids=Q49892&format=json&languages=en
3. 然后在根据mapping将Qid映射回Mid
"""


def main():
  with open(full_train_response_path, encoding='utf-8') as f:
    lines = f.readlines()
    with open(train_entity_recognition_failed_list_path, 'w', encoding='utf-8') as failed_wf:
      with open(train_qa_entities_pair_path, 'w', encoding='utf-8') as entities_wf:
        for line in lines:
          qa_pair = eval(line.strip())
          if 'entities' not in qa_pair['parsed_question']:
            failed_wf.write(line + '\n')
            failed_wf.flush()
            continue
          entities = qa_pair['parsed_question']['entities']
          useful_entities_info = []
          for entity in entities:
            useful_entity_info = {}
            if 'entityId' in entity:
              useful_entity_info['entityId'] = entity['entityId']
            else:
              continue
            if 'freebaseId' in entity:
              useful_entity_info['freebaseId'] = entity['freebaseId']
            if 'freebaseTypes' in entity:
              useful_entity_info['freebaseTypes'] = entity['freebaseTypes']

            if 'type' in entity:
              useful_entity_info['type'] = entity['type']
            if 'wikidataId' in entity:
              useful_entity_info['wikidataId'] = entity['wikidataId']
            useful_entities_info.append(useful_entity_info)
          new_qa_pair = {}
          new_qa_pair['question'] = qa_pair['question']
          new_qa_pair['ans'] = qa_pair['ans']
          new_qa_pair['entities'] = useful_entities_info
          entities_wf.write(str(new_qa_pair) + '\n')
          entities_wf.flush()


if __name__ == '__main__':
  main()
