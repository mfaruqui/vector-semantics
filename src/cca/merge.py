import sys

wordVecDict = {}
for fileName in sys.argv[1:]:
    for line in open(fileName, 'r'):
        wordVec = line.strip().split()
        if wordVec[0] not in wordVecDict:
            wordVecDict[wordVec[0]] = wordVec[1:]
        else:
            for val in wordVec[1:]:
                wordVecDict[wordVec[0]].append(val)

for word, vals in wordVecDict.iteritems():
    print word,
    for val in vals:
        print val,
    print
