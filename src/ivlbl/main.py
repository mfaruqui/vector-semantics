import sys
import argparse

from learning_corpus import LearningCorpus
from support_functions import print_vectors

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile", type=str, default=None, help="Raw corpus file name")
    parser.add_argument("-r", "--rowcutoff", type=int, default=0, help="Min frequency for a word to be a row")
    parser.add_argument("-w", "--window", type=int, default=5, help="Size of the word context window (one side)")
    parser.add_argument("-o", "--outputfile", type=str, default='outFile', help="Output file name")
    parser.add_argument("-l", "--numdim", type=int, default=80, help="Num of dimensions required in the SVD")
    parser.add_argument("-n", "--noise", type=int, default=10, help="Num of noisy samples to be used")
                    
    args = parser.parse_args()

    test = LearningCorpus(args.rowcutoff, args.window, args.noise, args.numdim, args.inputfile)
    print_vectors(test.vocab, test.wordVectors, args.outputfile)
