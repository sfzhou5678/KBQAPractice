

a=[1,2,3]
b=[4,5,6]
for i in a,b:
  print(i)

d=[i for i in (a,b)]
a.append(b)
print(a)
print(d)