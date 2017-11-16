


wf=open('hello.txt','w')
a=123
b=3.32

s=','.join([str(a),str(b)])
print(s)
wf.write('%d  %f'%(a,b))
# wf.write("abc"+'\n')
# wf.write("abc"+'\n')