# -*- coding: utf-8 -*-
from src.preparing_data.getEntInfo import *


tr_failToGetEntType_path = log_path + '/tr_failToGetAnsEntType.log'
tr_ansEnt_withTypeMid_fixConErr_path = data_path + '/tr_ansEnt_withTypeMid_fixConErr.txt'
tr_ansEntNoMid_path = log_path + '/tr_ansEntNoMid.log'
tr_entNoMidRate_path = log_path + '/tr_entNoMidRate.log'


logger_ansEntNoMId = logging.getLogger('tr_ansEntNoMid')
fileH0 = logging.FileHandler(tr_ansEntNoMid_path)
# conH0 = logging.StreamHandler()
logger_ansEntNoMId.addHandler(fileH0)
# logger_ansEntNoMId.addHandler(conH0)
logger_ansEntNoMId.setLevel("INFO")

logger_rate = logging.getLogger('tr_entNoMidRate')
fileH1 = logging.FileHandler(tr_entNoMidRate_path)
# conH1 = logging.StreamHandler()
logger_rate.addHandler(fileH1)
# logger_rate.addHandler(conH1)
logger_rate.setLevel("INFO")

def fixConnectErr(tr_failToGetEntType, tr_ansEnt_withTypeMid_raw, tr_ansEnt_withTypeMid_fixConnectErr):
    f_failToGetEntType = open(tr_failToGetEntType, 'r')
    failLines = f_failToGetEntType.readlines()
    f_failToGetEntType.close()

    f_tr_ansEnt_withTypeMid_raw = open(tr_ansEnt_withTypeMid_raw, 'r')
    rawLines = f_tr_ansEnt_withTypeMid_raw.readlines()
    f_tr_ansEnt_withTypeMid_raw.close()

    f_tr_ansEnt_withTM_fixErr = open(tr_ansEnt_withTypeMid_fixConnectErr, 'w')

    for failInfo in failLines:
        failInfDict = eval(failInfo)
        if 'index' in failInfDict:
            index = failInfDict['index']
            if 'error' in failInfDict:
                if failInfDict['error'] != "Qid doesn't have type":
                    if 'Qid' in failInfDict:
                        Qid = failInfDict['Qid']
                        entInfoJson = getHtml(
                            "https://www.wikidata.org/w/api.php?action=wbgetentities&ids=" + Qid + "&format=json")
                        entInfoDict = json.loads(entInfoJson)
                        if 'entities' in entInfoDict:
                            if Qid in entInfoDict['entities']:
                                if "type" in entInfoDict['entities'][Qid]:
                                    type = entInfoDict['entities'][Qid]["type"]
                                else:
                                    type = "null"  # 'type' = 'null'表明搜索结果没有type
                                    errorDict = {"index": index, "answer": failInfDict['answer'],
                                                 "Qid": Qid, "error": "Qid doesn't have type"}
                                    logger_ansEntType.error(errorDict)
        rawLinesDictTemp = eval(rawLines[index])
        answers = rawLinesDictTemp['ans']
        for ans in answers:
            if ans['name'] == failInfDict['answer']:
                entities = ans['entities']
                for entity in entities:
                    if entity['Qid'] == Qid:
                        entity['type'] = type
                        break
        rawLines[index] = str(rawLinesDictTemp) + '\n'
    print "Connection Problem Fixed."
    for line in rawLines:
        f_tr_ansEnt_withTM_fixErr.write(line)
    f_tr_ansEnt_withTM_fixErr.close()

def someStatics(tr_ansEnt_withTypeMid):
    f_tr_ansEnt_withTypeMid = open(tr_ansEnt_withTypeMid, 'r')
    fullPairs = f_tr_ansEnt_withTypeMid.readlines()
    f_tr_ansEnt_withTypeMid.close()

    ansType = {}
    entNoMid_50 = 0
    entNoMid_70 = 0
    entNoMid_90 = 0
    ansQidTotalNum = 0
    quesQidTotalNum = 0
    for line in fullPairs:
        fullPairDict = eval(line)
        ansQidNum = 0

        ansNoMid_flag = True
        ansNoMidNum = 0
        entNoMidNum = 0

        ansHasEntity_flag = True

        # Count the total amount of answers Qid(or answer entities)
        # Count the amount of answer type kinds, and the amount of each kind of answer type
        # 统计这些问答对：在一个问题的所有答案中，有答案的所有entities Qid都找不到Mid,如果是多答案的问题，
        # 则还要统计这样的answer数目，以及这样的答案在该问题的所有答案里所占的比例
        # 统计每一个问答对中，找不到Mid的实体占该问答对中所有答案实体的比例，给出超过50%，70% 和90% 的比例
        ansNoMid = {'index': fullPairDict['lineNum'], 'answers': []}
        entNoMid = {'index': fullPairDict['lineNum']}
        if 'ans' in fullPairDict:
            for answer in fullPairDict['ans']:
                if 'entities' in answer:
                    if not answer['entities']:
                        ansHasEntity_flag = False
                        break
            if ansHasEntity_flag:  # 去掉那些答案找不到Qid的问答对
                for answer in fullPairDict['ans']:
                    if 'entities' in answer:
                        ansQidNum += len(answer['entities'])
                        for entity in answer['entities']:
                            if 'type' in entity:
                                if entity['type'] in ansType:
                                    ansType[entity['type']] += 1
                                else:
                                    ansType[entity['type']] = 1
                            if 'Mid' in entity:
                                if entity['Mid'] != 'null':
                                    ansNoMid_flag = False
                                else:
                                    entNoMidNum += 1
                    if ansNoMid_flag:
                        ansNoMidNum += 1
                        ansNoMid['answers'].append(answer['name'])
                if ansNoMid_flag:
                    ansNoMid['wrongAnsNum'] = ansNoMidNum
                    ansNoMid['ansNum'] = len(fullPairDict['ans'])
                    ansNoMid['wrAnsRatio'] = 1.0 * ansNoMidNum / len(fullPairDict['ans'])
                    logger_ansEntNoMId.info(str(ansNoMid))

                entNoMidRatio = 1.0 * entNoMidNum / ansQidNum
                entNoMid['entNoMidRatio'] = entNoMidRatio
                logger_rate.info(str(entNoMid))

                ansQidTotalNum += ansQidNum

                if 0.5 <= entNoMidRatio < 0.7:
                    entNoMid_50 += 1
                elif 0.7 <= entNoMidRatio < 90:
                    entNoMid_70 += 1
                elif entNoMidRatio >= 0.9:
                    entNoMid_90 += 1
                # Count the total amount of question Qid(or question entities)
                if 'parsed_question' in fullPairDict:
                    if 'entities' in fullPairDict['parsed_question']:
                        quesQidTotalNum += len(fullPairDict['parsed_question']['entities'])

    print "Amount of ansQid: ",
    print ansQidTotalNum
    print "Amount of quesQid: ",
    print quesQidTotalNum
    print "Answers Type:",
    print ansType

    print "entNoMid_50: ",
    print entNoMid_50
    print "entNoMid_70: ",
    print entNoMid_70
    print "entNoMid_90: ",
    print entNoMid_90


def main():
    # fixConnectErr(tr_failToGetEntType_path, tr_ansEnt_withTypeMid_raw_path, tr_ansEnt_withTypeMid_fixConErr_path)
    someStatics(tr_ansEnt_withTypeMid_fixConErr_path)


if __name__ == '__main__':
    main()