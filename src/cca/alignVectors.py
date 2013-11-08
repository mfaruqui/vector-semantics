import sys
import argparse
import numpy

from alignments import read_word_vectors
from alignments import get_aligned_vectors

def save_orig_subset_and_aligned(outFileName, lang2WordVectors, lang1AlignedVectors):
    
    outFile = open(outFileName+'_orig_subset.txt','w')
    for word in lang1AlignedVectors:
        outFile.write(word+' '+' '.join([str(val) for val in lang2WordVectors[word]])+'\n')
    outFile.close()
    
    outFile = open(outFileName+'_new_aligned.txt','w')
    for word in lang1AlignedVectors:
        outFile.write(word+' '+' '.join([str(val) for val in lang1AlignedVectors[word]])+'\n')
    outFile.close()

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--wordaligncountfile", type=str, help="Word alignment count file")
    parser.add_argument("-w1", "--wordproj1", type=str, help="Word proj of lang1")
    parser.add_argument("-w2", "--wordproj2", type=str, help="Word proj of lang2")
    parser.add_argument("-o", "--outputfile", type=str, help="Output file for storing aligned vectors")
    
    args = parser.parse_args()
    lang1WordVectors = read_word_vectors(args.wordproj1)
    lang2WordVectors = read_word_vectors(args.wordproj2)
    
    lang1AlignedVectors = get_aligned_vectors(args.wordaligncountfile, lang1WordVectors, lang2WordVectors)
    save_orig_subset_and_aligned(args.outputfile, lang2WordVectors, lang1AlignedVectors)
