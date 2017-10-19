
def func(word_vocab,item_vocab,realtion_vocab):

  with open(r'G:\WikiData\ForFun\train.both.triples+candidate.txt') as f:
    for i in range(500):
      line = f.readline().strip()
      if line == '':
        break
      data = eval(line)
      question = data['question']
      topic_entity_id = data['topic_entity']
      forward_ans = data['forward_ans']
      forward_candicate_ans = data['forward_candidate_ans']

      reverse_ans = data['reverse_ans']
      reverse_candidate_ans = data['reverse_candidate_ans']

      print(len(forward_candicate_ans), len(reverse_candidate_ans))
