import random
import re
import string

data = 'hello, who are you?'

punctuation = string.punctuation
c = re.sub(r'[{}]+'.format(punctuation), ' ', data)
print(c)
