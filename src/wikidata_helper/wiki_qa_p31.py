import os
import json

from src.tools.common_tools import pickle_load, pickle_dump


def get_entity_set(data_path, entity_set_save_path):
  try:
    entity_set = pickle_load(entity_set_save_path)
    return entity_set
  except:
    pass
  entity_set = set()

  line_count = 0
  with open(data_path) as f:
    while True:
      line = f.readline().strip()
      if line == '':
        break
      line_count += 1
      if line_count % 100000 == 0:
        print(line_count, len(entity_set))
      q_item = json.loads(line)
      qid = q_item['id']
      entity_set.add(qid)  # 添加头QID

      for predicate in q_item['claims']:
        entity_set.add(predicate)  # 添加关系PID
        relevant_entities = q_item['claims'][predicate]
        for revelant_entity in relevant_entities:
          try:
            main_snak = revelant_entity['mainsnak']
            if main_snak['datavalue']['type'] == 'wikibase-entityid':
              revelant_qid = main_snak['datavalue']['value']['id']
              entity_set.add(revelant_qid)  # 添加尾QID
          except:
            pass
  pickle_dump(entity_set, entity_set_save_path)
  return entity_set


def get_p31(raw_wikidata_path, entity_set, res_path, error_path):
  error_wf = open(error_path, 'w')
  wf = open(res_path, 'w')

  total_line_count = 0
  in_entity_set_line_count = 0
  no_p31_count = 0
  with open(raw_wikidata_path) as f:
    f.readline()  # 由于第一行是一个'[' ,所以把他跳过
    while True:
      line = f.readline().strip()
      if line == ']' or line == '':
        # 最后一行是一个']'
        break

      if line.endswith(','):
        line = line[:-1]

      total_line_count += 1
      if total_line_count % 100000 == 0:
        print(total_line_count, in_entity_set_line_count, no_p31_count)
        wf.flush()
      q_item = json.loads(line)
      qid = q_item['id']
      if qid not in entity_set:
        continue

      in_entity_set_line_count += 1

      try:
        del q_item['sitelinks']
      except:
        pass

      if 'P31' not in q_item['claims']:
        no_p31_count += 1
        error_wf.write(line + '\n')
        error_wf.flush()
        pass
      else:
        claims = {}
        claims['P31'] = q_item['claims']['P31']
        q_item['claims'] = claims

        wf.write(json.dumps(q_item) + '\n')
  error_wf.close()
  wf.close()


def main():
  data_folder = r'F:\WikiData'
  raw_wikidata_path = os.path.join(data_folder, 'latest-all.json')

  selected_only_relevant_data_path = os.path.join(data_folder, 'SimpleQeustionBased',
                                                  'selected-latest-all.OnlyRelevant.data')
  entity_set_save_path = os.path.join(data_folder, 'SimpleQeustionBased', 'OnlyRelevant.entitySet.set')

  p31_res_path = os.path.join(data_folder, 'SimpleQeustionBased', 'p31.data')
  p31_error_path = os.path.join(data_folder, 'SimpleQeustionBased', 'p31Error.log')

  # 开始处理
  entity_set = get_entity_set(selected_only_relevant_data_path, entity_set_save_path)
  get_p31(raw_wikidata_path, entity_set, p31_res_path, p31_error_path)


if __name__ == '__main__':
  main()
