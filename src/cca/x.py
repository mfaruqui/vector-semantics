import sys

for line in sys.stdin:
    word = line.strip().split()[0]
    print word, '|||', word+'__1'
