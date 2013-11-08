import sys
import gzip
from collections import Counter
from operator import itemgetter

wordDict = Counter()
for line in gzip.open('/cab1/corpora/gigaword-v4.filtered/1m-sents.sample.txt.gz','r'):
    line = unicode(line,'utf-8')
    for word in line.strip().split():
        wordDict[word] += 1

for word, freq in sorted(wordDict.items(), key=itemgetter(1), reverse=True):
    print word.encode('utf-8'), freq
