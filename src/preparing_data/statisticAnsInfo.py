# Author: Duankang Fu
# Compiler: Anaconda 2

from src.preparing_data.getAnsInfo import *

train_ansEntity_fixConnectErr_path = data_path + '/trains_ansEntity_fixConnectErr.txt'

tr_oneEnt_path = data_path + '/tr_oneEnt.txt'
tr_multiEnt_path = data_path + '/tr_multiEnt.txt'

tr_oneAns_path = data_path + '/tr_oneAns.txt'
tr_multiAns_path = data_path + '/tr_multiAns.txt'

tr_ansEntFound_oneAns_path = data_path + '/tr_ansEntFound_oneAns.txt'
tr_ansNoEnt_oneAns_path = data_path + '/tr_ansEntNotFound_oneAns.txt'
tr_ansEntFound_multiAns_path = data_path + '/tr_ansEntFound_multiAns.txt'
tr_ansNoEnt_multiAns_path = data_path + '/tr_ansEntNotFound_multiAns.txt'

# Function: divideByEntitiesNum
# Description: If some entities failed to be searched when running getAnsInfo.getAnswersEntity()
#              due to connection timed out error, get them here.
def fixConnectionError(train_ansEntity_raw, train_connectionErr, train_ansEntity_fixErr):
    f_train_ansEntity_raw = open(train_ansEntity_raw, 'r')
    fullPairs = f_train_ansEntity_raw.readlines()
    f_train_ansEntity_raw.close()

    f_train_connectionErr = open(train_connectionErr, 'r')
    errorPairs = f_train_connectionErr.readlines()
    f_train_connectionErr.close()

    lenErrorPairs = len(errorPairs)

    f_train_ansEntity_fixErr = open(train_ansEntity_fixErr, 'w')

    for i in range(lenErrorPairs):
        errorLineDict = eval(errorPairs[i].strip())  # eval(): valuate the source in the context of globals and locals.
        # print str(errorLineDict)
        if 'error' in errorLineDict:
            if errorLineDict['error'] == "URLError(timeout('timed out',),)" or errorLineDict['error'] == "URLError(SSLError('_ssl.c:645: The handshake operation timed out',),)":
                if 'answer' in errorLineDict:
                    entities = []
                    ans = errorLineDict['answer']
                    ansTrans = ans.replace(" ", "%20")
                    entityJson = getHtml(
                        "https://www.wikidata.org/w/api.php?action=wbsearchentities&search=" + ansTrans + "&language=en&format=json")
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
                            entity["description"] = info["description"]
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
                    if 'index' in errorLineDict:
                        index = errorLineDict['index']
                        fullPairDictBuff = eval(fullPairs[index])
                        fullPairDictBuff['ans'].append({'entities': entities, 'name': ans})
                        fullPairs[index] = str(fullPairDictBuff) + '\n'
    print("Error fix finished.")
    for line in fullPairs:
        f_train_ansEntity_fixErr.write(line)
        # f_train_ansEntity_fixErr.write('\n')
    f_train_ansEntity_fixErr.close()


# Function: findNoEntity
# Description: Find all of the QAPairs where one or more answers has no entity.
def findNoEntity(train_ansEntity_fixErr, tr_ansNoEnt_oneAns,
                 tr_ansNoEnt_multiAns):
    f_train_ansEntity_fixErr = open(train_ansEntity_fixErr, 'r')
    fullPairs = f_train_ansEntity_fixErr.readlines()
    f_train_ansEntity_fixErr.close()

    f_tr_ansNoEnt_oneAns = open(tr_ansNoEnt_oneAns, 'w')
    f_tr_ansNoEnt_multiAns = open(tr_ansNoEnt_multiAns, 'w')

    for line in fullPairs:
        fullPairDict = eval(line.strip())
        if 'ans' in fullPairDict:
            AnsNum = len(fullPairDict['ans'])
            for answer in fullPairDict['ans']:
                if 'entities' in answer:
                    if not answer['entities']:
                        if AnsNum == 1:
                            f_tr_ansNoEnt_oneAns.write(line)
                        elif AnsNum > 1:
                            f_tr_ansNoEnt_multiAns.write(line)

    f_tr_ansNoEnt_oneAns.close()
    f_tr_ansNoEnt_multiAns.close()
    print "findNoEntity finished."

# Function: divideByEntitiesNum
# Once an answer of a question has more than one entity, save it in tr_multiEnt.
def divideByEntitiesNum(train_ansEntity_fixErr, tr_oneEnt, tr_multiEnt):
    f_train_ansEntity_fixErr = open(train_ansEntity_fixErr, 'r')
    fullPairs = f_train_ansEntity_fixErr.readlines()
    f_train_ansEntity_fixErr.close()

    f_tr_oneEnt = open(tr_oneEnt, 'w')
    f_tr_multiEnt = open(tr_multiEnt, 'w')

    oneEntFlag = True
    for line in fullPairs:
        fullPairDict = eval(line.strip())
        if 'ans' in fullPairDict:
            for answer in fullPairDict['ans']:
                if 'entities' in answer:
                    if len(answer['entities']) > 1:
                        oneEntFlag = False
                        break
            if oneEntFlag:
                f_tr_oneEnt.write(line)
            elif not oneEntFlag:
                f_tr_multiEnt.write(line)
    f_tr_oneEnt.close()
    f_tr_multiEnt.close()
    print "divideByEntitiesNum finished."

# Divide one-answer question and multi_answer question
def divideByAnsNum(train_ansEntity_fixErr, tr_oneAns, tr_multiAns):
    f_train_ansEntity_fixErr = open(train_ansEntity_fixErr, 'r')
    fullPairs = f_train_ansEntity_fixErr.readlines()
    f_train_ansEntity_fixErr.close()

    f_tr_oneAns = open(tr_oneAns, 'w')
    f_tr_multiAns = open(tr_multiAns, 'w')

    for line in fullPairs:
        fullPairDict = eval(line.strip())
        if 'ans' in fullPairDict:
            if len(fullPairDict['ans']) > 1:
                f_tr_multiAns.write(line)
            elif len(fullPairDict['ans']) == 1:
                f_tr_oneAns.write(line)
    f_tr_oneAns.close()
    f_tr_multiAns.close()
    print "divideByAnsNum finished."

def main():
    fixConnectionError(train_ansEntity_raw_path, train_failToGetEntity_path, train_ansEntity_fixConnectErr_path)
    findNoEntity(train_ansEntity_fixConnectErr_path, tr_ansNoEnt_oneAns_path, tr_ansNoEnt_multiAns_path)
    divideByEntitiesNum(train_ansEntity_fixConnectErr_path, tr_oneEnt_path, tr_multiEnt_path)
    divideByAnsNum(train_ansEntity_fixConnectErr_path, tr_oneAns_path, tr_multiAns_path)


if __name__ == "__main__":
    main()