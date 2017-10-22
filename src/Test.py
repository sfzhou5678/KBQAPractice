# -*- coding: utf-8 -*-

data_path = "../../data"

tr_re_path = data_path + '/webquestions.train.textrazor.full.txt'

def getLine(file_path):
    f_open = open(file_path,'r')
    lines = f_open.readlines()

    len = len(lines)

    for i in range(100):
        lineDict = eval(lines[i].strip())
        if 'question' in lineDict:
            print str(lineDict['question']) + '\n'

def main():
    getLine(tr_re_path)


if __name__ == '__main__':
    main()


