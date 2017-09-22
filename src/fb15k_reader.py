import os


def main():
  data_folder = r'C:\Users\zsf\Desktop\FB15K-237.2\Release'
  train_path = os.path.join(data_folder, 'train.txt')
  valid_path = os.path.join(data_folder, 'valid.txt')
  test_path = os.path.join(data_folder, 'test.txt')

  train_e_dic = set()
  train_r_dic = set()
  with open(train_path) as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip()
      h, r, t = line.split('\t')
      train_e_dic.add(h)
      train_e_dic.add(t)

      train_r_dic.add(r)

  valid_e_dic = set()
  valid_r_dic = set()
  with open(valid_path) as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip()
      h, r, t = line.split('\t')
      valid_e_dic.add(h)
      valid_e_dic.add(t)

      valid_r_dic.add(r)

  test_e_dic = set()
  test_r_dic = set()
  with open(test_path) as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip()
      h, r, t = line.split('\t')
      test_e_dic.add(h)
      test_e_dic.add(t)

      test_r_dic.add(r)

  print(len(train_r_dic))
  print(len(valid_r_dic))
  print(len(test_r_dic))

  total_r_dic = set()
  for item in train_r_dic:
    total_r_dic.add(item)
  for item in test_r_dic:
    total_r_dic.add(item)
  for item in valid_r_dic:
    total_r_dic.add(item)

  with open('../data/relations.txt', 'w') as wf:
    for item in total_r_dic:
      wf.write(item + '\n')


if __name__ == '__main__':
  main()
