# -*- coding: utf-8 -*-
from src.preparing_data.getEntInfo import *

tr_failToGetEntType_path = log_path + '/tr_failToGetAnsEntType.log'
tr_ansEnt_withTypeMid_fixConErr_path = data_path + '/tr_ansEnt_withTypeMid_fixConErr.txt'
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


fixConnectErr(tr_failToGetEntType_path, tr_ansEnt_withTypeMid_raw_path, tr_ansEnt_withTypeMid_fixConErr_path)