import os
import pickle

data_path = '../../data/train_ansEntity.txt'


def get_Q_M_mapping():
  data_folder = r'D:\MeachineLearningData'
  mapping_path = os.path.join(data_folder, 'fb2w.nt')

  mapping = {}
  with open(mapping_path) as f:
    lines = f.readlines()
    for line in lines:
      try:
        line = line.strip()
        MID, _, QID = line[:-2].split('\t')
        mapping[QID] = MID
      except:
        print(line)
  return mapping


# QID_to_MID_mapping = {}
QID_to_MID_mapping = get_Q_M_mapping()

print(len(QID_to_MID_mapping))

freebase_MID = set()
with open(data_path) as f:
  lines = f.readlines()
  print(len(lines))

  has_entities_count = 0
  no_QID_mapping_count = 0
  for line in lines:
    qa_pair = eval(line.strip())

    # 提取question中的MID
    if 'entities' in qa_pair['parsed_question']:
      has_entities_count += 1
      entities = qa_pair['parsed_question']['entities']
      for entity in entities:
        if 'freebaseId' in entity:
          try:
            MID = entity['freebaseId']
            if MID[2] == '/':
              MID = '/m.' + MID[3:]
            else:
              try:
                pos = MID[1:].index('/')
                MID = '/m.' + MID[pos + 2:]
              except:
                print('%s 中不存在/' % MID)
                continue
            MID = '<http://rdf.freebase.com/ns' + MID + '>'
            freebase_MID.add(MID)
          except:
            try:
              QID = entity['wikidataId']
              QID = '<http://www.wikidata.org/entity/' + QID + '>'
              MID = QID_to_MID_mapping[QID]
              freebase_MID.add(MID)
            except:
              print('error entity: %s' % str(entity))

    # 提取ans中的MID
    ans = qa_pair['ans']
    for a in ans:
      if 'entities' in a:
        entities = a['entities']
        for entity in entities:
          try:
            QID = entity['Qid']
            QID = '<http://www.wikidata.org/entity/' + QID + '>'
            MID = QID_to_MID_mapping[QID]
            freebase_MID.add(MID)
          except:
            no_QID_mapping_count += 1

  print(has_entities_count)
  print(no_QID_mapping_count)
  print(len(freebase_MID))

mid_write = open('../../data/WebQuestion.mid.pkl', 'wb')
pickle.dump(freebase_MID, mid_write, True)
