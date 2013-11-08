import sys
import argparse
import numpy

from alignments import read_word_vectors
from alignments import get_aligned_vectors

def save_orig_subset_and_aligned(outFileName, lang2WordVectors, alignedVectors):
    
    outFile = open(outFileName+'_orig_subset.txt','w')
    for word in alignedVectors:
        outFile.write(word+' '+' '.join([str(val) for val in lang2WordVectors[word]])+'\n')
    outFile.close()
    
    outFile = open(outFileName+'_new_aligned.txt','w')
    for word in alignedVectors:
        outFile.write(word+' '+' '.join([str(val) for val in alignedVectors[word]])+'\n')
    outFile.close()

def get_all_lang_pairs_alignments(args):

    lang1WordVectors = read_word_vectors(args.wordproj1)
    lang2WordVectors = read_word_vectors(args.wordproj2)

    lang1AlignedVectors = get_aligned_vectors(args.wordaligncountfile12, lang1WordVectors, lang2WordVectors)

    print len(lang1AlignedVectors)

    lang3WordVectors = read_word_vectors(args.wordproj3)
    lang3AlignedVectors = get_aligned_vectors(args.wordaligncountfile32, lang3WordVectors, lang2WordVectors)

    print len(lang3AlignedVectors)

    lang4WordVectors = read_word_vectors(args.wordproj4)
    lang4AlignedVectors = get_aligned_vectors(args.wordaligncountfile42, lang4WordVectors, lang2WordVectors)

    print len(lang4AlignedVectors)

    ''' Create a final aligned vector which contains all dimensions from different aligned vectors '''
    commonWords = set(lang1AlignedVectors.keys()) & set(lang3AlignedVectors) & set(lang4AlignedVectors)
    randWord = list(commonWords)[0]
    numFinalDims = len(lang1AlignedVectors[randWord]) + len(lang3AlignedVectors[randWord]) + len(lang4AlignedVectors[randWord]) 

    print len(commonWords)
    
    alignedVectors = {}
    for word in commonWords:
        index = 0
        alignedVectors[word] = numpy.zeros((numFinalDims), dtype=float)
        for val in lang1AlignedVectors[word]:
            alignedVectors[word][index] += val
            index += 1

        for val in lang3AlignedVectors[word]:
            alignedVectors[word][index] += val
            index += 1

        for val in lang4AlignedVectors[word]:
            alignedVectors[word][index] += val
            index += 1

    return alignedVectors, lang2WordVectors
    
if __name__=='__main__':

    ''' Lang2 is English (or the bridge language)'''
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-a12", "--wordaligncountfile12", type=str, help="Word alignment count file of lang1 and 2")
    parser.add_argument("-a32", "--wordaligncountfile32", type=str, default=None, help="Word alignment count file of lang3 and 2")
    parser.add_argument("-a42", "--wordaligncountfile42", type=str, default=None, help="Word alignment count file of lang4 and 2")
    parser.add_argument("-w1", "--wordproj1", type=str, help="Word proj of lang1")
    parser.add_argument("-w2", "--wordproj2", type=str, help="Word proj of lang2")
    parser.add_argument("-w3", "--wordproj3", type=str, help="Word proj of lang3")
    parser.add_argument("-w4", "--wordproj4", type=str, help="Word proj of lang4")
    parser.add_argument("-o", "--outputfile", type=str, help="Output file for storing aligned vectors")
    
    args = parser.parse_args()
    
    alignedVectors, lang2WordVectors = get_all_lang_pairs_alignments(args)
    save_orig_subset_and_aligned(args.outputfile, lang2WordVectors, alignedVectors)
