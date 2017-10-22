import random
import re
import string
import os
import numpy as np
import heapq

data=np.array([1,2,3,4,5])

top_5_ans_index = heapq.nlargest(3, range(len(data)), data.take)
print(top_5_ans_index)