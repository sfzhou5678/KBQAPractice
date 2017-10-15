# -*- coding: utf-8 -*-

# Author: Duankang Fu
# Compiler: Anaconda 2

import urllib2
import json
import threading
import logging

data_path = '../../data'
src_path = '../../src'
log_path = '../../log'

train_res_path = '../../data/webquestions.train.textrazor.full.txt'
test_res_path = '../../data/webquestions.test.textrazor.full.txt'

train_ansEntity_raw_path = data_path + '/train_ansEntity_raw.txt'
train_failToGetEntity_path = log_path + '/train_failToGetEntity.log'
ts_ansEntity_raw_path = data_path + '/ts_ansEntity_raw.txt'
ts_failToGetEntity_path = log_path + '/ts_failToGetEntity.log'

logger = logging.getLogger(__name__)
fileH = logging.FileHandler(ts_failToGetEntity_path)
conH = logging.StreamHandler()
logger.addHandler(fileH)
logger.addHandler(conH)



def getHtml(url):
    page = urllib2.urlopen(url, timeout=10)  # When the online API search time of an answer exceeds 10 sec, report it in the log file.
    html = page.read()
    return html


# Function: getAnswersEntity
# Description: Get the corresponding entities of each answers in each QA Pair.
# Note: A question might have several answers, and an answer might correspond to several entities.
#       TODO: Therefore, Semantic disambiguation is needed to get the right entity.
def getAnswersEntity(file_path):
    infile = open(file_path, 'r')
    lines = infile.readlines()

    lenPairs = len(lines)
    lenPart = lenPairs / 3
    f_tr_ansEntity_raw = open(ts_ansEntity_raw_path, 'w')
    lock = threading.Lock()

    t1 = threading.Thread(target=getAnswersEntity_thread,
                          args=(0, lenPart, lines,
                                f_tr_ansEntity_raw, lock))

    t2 = threading.Thread(target=getAnswersEntity_thread,
                          args=(lenPart, lenPart * 2, lines,
                                f_tr_ansEntity_raw, lock))

    t3 = threading.Thread(target=getAnswersEntity_thread,
                          args=(lenPart * 2, lenPairs, lines,
                                f_tr_ansEntity_raw, lock))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print "All threads ended."

# Function: getAnswersEntity_thread
# Description: Use multi-threads methods to fasten the process.
def getAnswersEntity_thread(begin, end, lines, f_tr_ansEntity, lock):
    for index in range(begin, end):  # Be aware that (index + 1) is the line number of a piece of full QAPairs information.
        pairsFullInfo = eval(lines[index].strip())
        answersEntities = []
        for ans in pairsFullInfo['ans']:
            entities = []
            ansTrans = ans.replace(" ", "%20")
            try:
                entityJson = getHtml("https://www.wikidata.org/w/api.php?action=wbsearchentities&search="+ansTrans+"&language=en&format=json")
                entityDict = json.loads(entityJson)
            except Exception, e:
                if repr(e) == "ValueError('No JSON object could be decoded',)":
                    answersEntities.append({"name": ans, "entities": entities})
                else:
                    logger.error(str(pairsFullInfo))
                continue
            entityInfo = entityDict["search"]
            for info in entityInfo:
                entity = {"Qid": info["id"]}
                # Some entity may not consists of "label" or "description" property
                if "label" in info:
                    entity["label"] = info["label"]
                else:
                    entity["label"] = "null"
                if "description" in info:
                    entity["description"] = info['description']
                else:
                    entity["description"] = "null"
                if 'url' in info:
                    entity["url"] = info["url"]
                else:
                    entity["url"] = "null"
                if "concepturi" in info:
                    entity["concepturi"] = info["concepturi"]
                else:
                    entity["concepturi"] = "null"
                if "pageid" in info:
                    entity["pageid"] = info["pageid"]
                else:
                    entity["pageid"] = "null"
                entities.append(entity)
            answersEntities.append({"name": ans, "entities": entities})
        pairsFullInfo['ans'] = answersEntities
        # Because multi threads method are used, the order where the file print is changed. So I
        # record the line number of a piece of full QAPairs information in test_res_path.
        pairsFullInfo['lineNum'] = index
        lock.acquire()
        f_tr_ansEntity.write(str(pairsFullInfo))
        # print str(pairsFullInfo)
        f_tr_ansEntity.write('\n')
        lock.release()

# Function: getAnswersEntity_direct
# Description: Using normal method rather than multi-threads.
def getAnswersEntity_direct(file_path):
    infile = open(file_path, 'r')
    lines = infile.readlines()

    lenPairs = len(lines)
    f_tr_ansEntity_raw = open(train_ansEntity_raw_path, 'w')

    for index in range(lenPairs):  # Be aware that (index + 1) is the line number of a piece of full QAPairs information.
        pairsFullInfo = eval(lines[index].strip())
        answersEntities = []
        for ans in pairsFullInfo['ans']:
            entities = []
            ansTrans = ans.replace(" ", "%20")
            entityJson = getHtml("https://www.wikidata.org/w/api.php?action=wbsearchentities&search="+ansTrans+"&language=en&format=json")
            entityDict = json.loads(entityJson)
            entityInfo = entityDict["search"]
            for info in entityInfo:
                entity = {"Qid": info["id"]}
                # Some entity may not consists of "label" or "description" property
                if "label" in info:
                    entity["label"] = info["label"]
                else:
                    entity["label"] = "null"
                if "description" in info:
                    entity["description"] = info['description']
                else:
                    entity["description"] = "null"
                if 'url' in info:
                    entity["url"] = info["url"]
                else:
                    entity["url"] = "null"
                if "concepturi" in info:
                    entity["concepturi"] = info["concepturi"]
                else:
                    entity["concepturi"] = "null"
                if "pageid" in info:
                    entity["pageid"] = info["pageid"]
                else:
                    entity["pageid"] = "null"
                entities.append(entity)
            answersEntities.append({"name": ans, "entities": entities})
        pairsFullInfo['ans'] = answersEntities
        # Because multi threads method are used, the order where the file print is changed. So I
        # record the line number of a piece of full QAPairs information in test_res_path.
        pairsFullInfo['lineNum'] = index
        f_tr_ansEntity_raw.write(str(pairsFullInfo))
        # print str(pairsFullInfo)
        f_tr_ansEntity_raw.write('\n')

def main():
    getAnswersEntity(test_res_path)
    print "Back to main."


if __name__ == '__main__':
    main()
