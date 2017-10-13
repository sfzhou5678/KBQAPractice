# -*- coding: utf-8 -*-

from statisticEntInfo import *

tr_ansHasEnt_oneAns_path = data_path + '/tr_ansHasEnt_oneAns.txt'
tr_oneAnsSubSet_path = data_path + "/tr_oneAnsSubSet_oneMid.txt"

def findHasEntity(tr_ansEntity, tr_ansHasEnt_oneAns):
    f_tr_ansEntity_ = open(tr_ansEntity, 'r')
    fullPairs = f_tr_ansEntity_.readlines()
    f_tr_ansEntity_.close()

    f_tr_ansHasEnt_oneAns = open(tr_ansHasEnt_oneAns, 'w')

    for line in fullPairs:
        fullPairDict = eval(line.strip())
        if 'ans' in fullPairDict:
            AnsNum = len(fullPairDict['ans'])
            for answer in fullPairDict['ans']:
                if 'entities' in answer:
                    if answer['entities']:
                        if AnsNum == 1:
                            f_tr_ansHasEnt_oneAns.write(line)

    f_tr_ansHasEnt_oneAns.close()
    print "findNoEntity finished."
def getOneAnsSubSet(tr_ansEnt, outFileName, ratio):
    infile = open(tr_ansEnt, 'r')
    lines = infile.readlines()
    infile.close()

    outfile = open(outFileName, 'w')

    for line in lines:
        QAInfoDict = eval(line.strip())
        if 'ans' in QAInfoDict:
            # There is only one answer here.
            answer = QAInfoDict['ans']
            if len(answer) > 1:
                print "Error! Multi-answer QA pairs found!"
                exit()
            # get rid of QA pairs whose answer's entities are not found.
            if 'entities' in answer[0]:  # answer is a list and answer[0] is a dictionary
                if answer[0]['entities']:
                    # get rid of QA pairs where answer entities who has MID account for less than 50% of all entities
                    entities = answer[0]['entities']
                    entityNum = len(entities)
                    hasMidNum = 0
                    for entity in entities:
                        if entity['Mid'] == 'null':
                            hasMidNum += 1
                    if 1.0 * hasMidNum / entityNum >= ratio:
                        outfile.write(line)

    outfile.close()
    print "getOneAnsSubSet finished."

def getOneAnsSubSet2(tr_ansEnt, outFileName):
    infile = open(tr_ansEnt, 'r')
    lines = infile.readlines()
    infile.close()

    outfile = open(outFileName, 'w')

    for line in lines:
        QAInfoDict = eval(line.strip())
        if 'ans' in QAInfoDict:
            # There is only one answer here.
            answer = QAInfoDict['ans']
            if len(answer) > 1:
                print "Error! Multi-answer QA pairs found!"
                exit()
            # get rid of QA pairs whose answer's entities are not found.
            if 'entities' in answer[0]:  # answer is a list and answer[0] is a dictionary
                if answer[0]['entities']:
                    # get rid of QA pairs where answer entities who has MID account for less than 50% of all entities
                    entities = answer[0]['entities']
                    for entity in entities:
                        if entity['Mid'] != 'null':
                            QAInfoDict['ans'][0]['entities'] = [entity]
                            outfile.write(str(QAInfoDict) + '\n')
                            break
    outfile.close()

def main():
    # findHasEntity(tr_ansEnt_withTypeMid_fixConErr_path, tr_ansHasEnt_oneAns_path)
    getOneAnsSubSet2(tr_ansHasEnt_oneAns_path, tr_oneAnsSubSet_path)
if __name__ == '__main__':
    main()



