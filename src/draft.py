import heapq

n_item=[1]
item='abcccdsaf'
for char in list(item[:3]):
  # fixme: unk的ID还未确定
  n_item.append(char)
  n_item.append('End')
print(n_item)