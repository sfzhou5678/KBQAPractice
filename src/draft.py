import heapq
import numpy as np

datas=np.reshape(range(4*10*20),[4,10,20])
pos=np.reshape(range(4),[4])

print(datas)
print(pos)

for i in range(4):
  p=pos[i]
  print(datas[i,p])

# print(datas[pos].shape)
# print(np.take(datas,pos).shape)
# 期望选出的是[4,20]

