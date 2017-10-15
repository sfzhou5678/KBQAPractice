import json

"""

#### 1 hop：

Q: 'what is the name of justin bieber brother?'
TOPIC: 'justin bieber' wikiID:'Q34086'

ans: ['Jazmyn Bieber', 'Jaxon Bieber']
QID:[Q27801043,Q27801041]

[Q34086-claims-[P3373(sibling)]-mainsnak-[0]datavalue-value-id=Q27801043]
[Q34086-claims-[P3373(sibling)]-mainsnak-[1]datavalue-value-id=Q27801041]

==============================================================================
Q:'what kind of money to take to bahamas?'
topic: 'The Bahamas' 'Q778'
ans:'Bahamian dollar' Q194339

{Q778-claims-[P17(country)]-mainsnak-[0]datavalue-value-id=Q778}

==============================================================================
q: 'what character did john noble play in lord of the rings?'
t:'john noble' 'Q312399'
a:'Denethor II' Q718380

Path: Q -> P175(performer) -> Q718380


==============================================================================
Q2: 'what character did natalie portman play in star wars?'
topic：'Natalie Portman' 'Q37876'
topic2：'Star Wars' 'Q462'

ans: 'Padmé Amidala'
QID:Q51789


==============================================================================
Q3:'what country is the grand bahama island in?'
topic：'Grand Bahama' 'Q866345'
ans：'Bahama' 'Q4842313'

"""

with open(r'C:\Users\hasee\Desktop\jb.txt') as f:
  str = f.readline().strip()
justin_biber_obj = str
QID = 'Q34086'
obj = json.loads(justin_biber_obj)['entities'][QID]

claims = obj['claims']
main_snak_keys = set()
main_snak_datatype = set()
main_snak_datavalue_type = set()

for p in claims:
  revelant_entities = claims[p]
  for entity in revelant_entities:
    datatype = entity['mainsnak']['datavalue']['type']
    datavalue = entity['mainsnak']['datavalue']['value']
    print(datatype, datavalue)
    if datatype == 'wikibase-entityid':
      # wikibase-entityid {'numeric-id': 148, 'id': 'Q148', 'entity-type': 'item'}
      target_qid = datavalue['id']
    elif datatype == 'string':
      # string 031767702 | string grid.8547.e
      pass
    elif datatype == 'time':
      # time {'before': 0, 'timezone': 0, 'time': '+1905-00-00T00:00:00Z', 'calendarmodel': 'http://www.wikidata.org/entity/Q1985727', 'precision': 9, 'after': 0}
      pass
    elif datatype == 'globalcoordinate':
      # globecoordinate {'globe': 'http://www.wikidata.org/entity/Q2', 'altitude': None, 'precision': None, 'latitude': 31.298888888889, 'longitude': 121.49916666667}
      pass

    elif datatype == 'monolingualtext':
      # monolingualtext {'language': 'en', 'text': 'Justin Bieber'}
      pass
    elif datatype == 'quantity':
      # quantity {'amount': '+1.75', 'unit': 'http://www.wikidata.org/entity/Q11573'}
      pass
    else:
      print('=====[%s]=====' % datatype)
    # 辅助工具：
    for key in entity['mainsnak']:
      main_snak_keys.add(key)

    main_snak_datatype.add(entity['mainsnak']['datatype'])  # 应该记录这个
    main_snak_datavalue_type.add(entity['mainsnak']['datavalue']['type'])  # 这个分的不够细，可以不用

print()
print(main_snak_keys)
print(main_snak_datatype)
print(main_snak_datavalue_type)

# TODO 分别拿出datavalue里面的type和mainsnak中的datatype，进行对比(现在不知道到底应该用哪个type，有可能这两个就是等价的type，所以需要统计一下)
