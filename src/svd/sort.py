import sys
from operator import itemgetter

phraseDict = {}
for line in sys.stdin:
     line = line.strip()
     word1, word2, rest = line.split(' ',2)
     phraseDict[(word1, word2)] = rest

for (w1, w2) in sorted(phraseDict.keys(), key=itemgetter(0), reverse=True):
    print w1, w2, phraseDict[(w1, w2)]
