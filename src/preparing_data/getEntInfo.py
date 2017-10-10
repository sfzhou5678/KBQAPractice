# -*- coding: utf-8 -*-

# Author: Duankang Fu
# Compiler: Anaconda 2

import json
import logging
from getAnsInfo import getHtml, data_path, log_path
import pickle

from statisticAnsInfo import train_ansEntity_fixConnectErr_path
tr_failToGetAnsEntMid_path = log_path + "/tr_failToGetAnsEntMid.log"
tr_failToGetAnsEntType_path = log_path + '/tr_failToGetAnsEntType.log'
tr_failToGetQEntMid_path = log_path + "/tr_failToGetQEntMid.log"

logger_ansEntMid = logging.getLogger("tr_failToGetAnsEntMid")
fileH1 = logging.FileHandler(tr_failToGetAnsEntMid_path)
conH1 = logging.StreamHandler()
logger_ansEntMid.addHandler(fileH1)
logger_ansEntMid.addHandler(conH1)

logger_ansEntType = logging.getLogger("tr_failToGetAnsEntType")
fileH2 = logging.FileHandler(tr_failToGetAnsEntType_path)
conH2 = logging.StreamHandler()
logger_ansEntType.addHandler(fileH2)
logger_ansEntType.addHandler(conH2)

logger_QEntMid = logging.getLogger("tr_failToGetQEntMid")
fileH3 = logging.FileHandler(tr_failToGetQEntMid_path)
conH3 = logging.StreamHandler()
logger_QEntMid.addHandler(fileH3)
logger_QEntMid.addHandler(conH3)

tr_ansEnt_withTypeMid_raw_path = data_path + '/tr_ansEnt_withTypeMid_raw.txt'

fb2w_path = data_path + '/fb2w.nt'
fb2wDict_pkl_path = data_path + '/fb2wDict.pkl'

def get_fb2wMapping(fb2w_path):
    fb2wMapping = {}
    with open(fb2w_path) as f_fb2w:
        fb2wTriples = f_fb2w.readlines()
        for triple in fb2wTriples:
            try:
                triple = triple.strip()
                MID, _, QID = triple[:-2].split('\t')
                fb2wMapping[QID] = MID
            except Exception:
                print(triple)
    f_fb2wDict_pkl = open(fb2wDict_pkl_path, 'w')
    pickle.dump(fb2wMapping, f_fb2wDict_pkl)

def getEntTypeAndMid(tr_ansEntity, tr_ansEnt_withTypeMid, fb2wDict_pkl_path):
    f_tr_ansEntity = open(tr_ansEntity, 'r')
    fullPairs = f_tr_ansEntity.readlines()
    f_tr_ansEntity.close()

    f_tr_ansEnt_withTypeMid = open(tr_ansEnt_withTypeMid, 'w')
    try:
        f_fb2wDict_pkl = open(fb2wDict_pkl_path, 'r')
        fb2wMapping = pickle.load(f_fb2wDict_pkl)
        f_fb2wDict_pkl.close()
    except:
        print "fb2wMapping doesn't exit."
        pass

    for line in fullPairs:
        fullPairDict = eval(line.strip())
        # Get Mid of entity in question
        if 'entities' in fullPairDict['parsed_question']:
            entities = fullPairDict['parsed_question']['entities']
            for entity in entities:
                if 'freebaseId' in entity:
                    try:
                        MID = entity['freebaseId']
                        if MID[2] == '/':
                            MID = '/m.' + MID[3:]
                        else:
                            try:
                                pos = MID[1:].index('/')
                                MID = '/m.' + MID[pos + 2:]
                            except:
                                errorDict = {"index": fullPairDict['linNum'], "Q_freebaseId": entity['freebaseId'],
                                             'QWikidataId': entity['wikidataId'],
                                             "error": "Question entity doesn't have MID originally.Go find it by trying fb2wMapping."}
                                logger_QEntMid.error(errorDict)
                                continue
                        MID = '<http://rdf.freebase.com/ns' + MID + '>'
                        entity['freebaseId'] = MID
                    except:
                        try:
                            QID = entity['wikidataId']
                            QID = '<http://www.wikidata.org/entity/' + QID + '>'
                            MID = fb2wMapping[QID]
                            entity['freebaseId'] = MID
                        except:
                            errorDict = {"index": fullPairDict['lineNum'], "QFreebaseId": entity['freebaseId'],
                                         'QWikidataId': entity['wikidataId'],
                                         "error": "Question entity freebaseId seems invalid."}
                            logger_QEntMid.error(errorDict)
        # Get entity type and entity Mid in answers.
        if 'ans' in fullPairDict:
            for answer in fullPairDict['ans']:
                if 'entities' in answer:
                    for entity in answer['entities']:
                        if 'Qid' in entity:
                            try:
                                entInfoJson = getHtml("https://www.wikidata.org/w/api.php?action=wbgetentities&ids="+entity['Qid']+"&format=json")
                                entInfoDict = json.loads(entInfoJson)
                                if 'entities' in entInfoDict:
                                    if entity['Qid'] in entInfoDict['entities']:
                                        if "type" in entInfoDict['entities'][entity['Qid']]:
                                            entity["type"] = entInfoDict['entities'][entity['Qid']]["type"]
                                        else:
                                            entity["type"] = "null"  # 'type' = 'null'表明搜索结果没有type
                                            errorDict = {"index": fullPairDict['lineNum'], "answer": answer['name'],
                                                         "Qid": entity['Qid'], "error": "Qid doesn't have type"}
                                            logger_ansEntType.error(errorDict)
                            except Exception, e1:
                                errorDict = {"index": fullPairDict['lineNum'], "answer": answer['name'],
                                             "Qid": entity['Qid'], "error":  repr(e1)}
                                logger_ansEntType.error(errorDict)
                            try:
                                QID = entity['Qid']
                                QID = '<http://www.wikidata.org/entity/' + QID + '>'
                                MID = fb2wMapping[QID]
                                entity["Mid"] = MID
                            except:
                                errorDict = {"index": fullPairDict['lineNum'], "answer": answer['name'],
                                             "Qid": entity['Qid'], "error": "Answer Qid2Mid mapping failed."}
                                entity["Mid"] = "null"  # 'Mid' = 'null'表明无法找到Qid到Mid的映射
                                logger_ansEntMid.error(errorDict)

        f_tr_ansEnt_withTypeMid.write(str(fullPairDict) + '\n')
    f_tr_ansEnt_withTypeMid.close()

def main():
    # get_fb2wMapping(fb2w_path)
    getEntTypeAndMid(train_ansEntity_fixConnectErr_path, tr_ansEnt_withTypeMid_raw_path, fb2wDict_pkl_path)


if __name__ == '__main__':
    main()

