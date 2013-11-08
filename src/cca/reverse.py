import sys
from operator import itemgetter

wordAlignDict = {}
for line in sys.stdin:
    en, fr_freqs = line.strip().split(' ||| ')
    for fr_freq in fr_freqs.split():
        fr, freq = fr_freq.rsplit('__',1)
        freq = int(freq)
        if fr in wordAlignDict:
            if en in wordAlignDict[fr]:
                wordAlignDict[fr][en] += freq
            else:
                wordAlignDict[fr][en] = freq
        else:
            wordAlignDict[fr] = {en: freq}

for word in wordAlignDict.iterkeys():
    print word, '|||',
    for alignedWord, freq in sorted(wordAlignDict[word].items(), key=itemgetter(1), reverse=True):
        print alignedWord+'__'+str(freq),
    print
