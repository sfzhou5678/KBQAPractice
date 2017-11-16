import heapq


num=0
def aaa(n):
  global  num
  n=num
  n+=1
  num+=1


def main():
  global num
  aaa(num)

  print(num)

if __name__ == '__main__':
    main()