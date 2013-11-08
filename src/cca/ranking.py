# rho = 1 - 6*d*d/(n*(n*n-1))
import math
from operator import itemgetter

def assign_ranks(itemDict):

    rankedDict = {}
    '''
    sortedList = [(key, val) for (key, val) in sorted(itemDict.items(), key=itemgetter(1), reverse=True)]
    for i, (key, val) in enumerate(sortedList):
        sameValIndices = [j for j, (key2, val2) in enumerate(sortedList) if val2 == val]
        if len(sameValIndices) == 1:
            rankedDict[key] = i
        else:
            avgRank = 1.*sum(sameValIndices)/len(sameValIndices)
            rankedDict[key] = avgRank
    '''

    rank = 0
    for key, val in sorted(itemDict.items(), key=itemgetter(1), reverse=True):
         rankedDict[key] = rank
         rank += 1

    return rankedDict

def spearmans_rho(rankedDict1, rankedDict2):

    assert len(rankedDict1) == len(rankedDict2)

    x_avg = sum([val for val in rankedDict1.values()])/len(rankedDict1)
    y_avg = sum([val for val in rankedDict2.values()])/len(rankedDict2)

    num = 0
    d_x = 0
    d_y = 0
    for key in rankedDict1.keys():
        xi = rankedDict1[key]
        yi = rankedDict2[key]

        num += (xi-x_avg)*(yi-y_avg)
        d_x += (xi-x_avg)**2
        d_y += (yi-y_avg)**2

    return 1.*num/(math.sqrt(d_x*d_y))
