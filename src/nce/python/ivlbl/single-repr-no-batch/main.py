import sys
import argparse
#import pyximport; pyximport.install()

from learning_corpus import LearningCorpus
from support_functions import print_vectors

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--inputfile", type=str, default=None, help="Raw corpus file name")
    parser.add_argument("-f", "--freqcutoff", type=int, default=0, help="Min frequency for a word to be a row")
    parser.add_argument("-w", "--window", type=int, default=5, help="Size of the word context window (one side)")
    parser.add_argument("-o", "--outputfile", type=str, default='outFile', help="Output file name")
    parser.add_argument("-l", "--numdim", type=int, default=80, help="Num of dimensions required in the SVD")
    parser.add_argument("-n", "--noise", type=int, default=10, help="Num of noisy samples to be used")
    parser.add_argument("-r", "--learningrate", type=float, default=0.5, help="Learning rate to be used")
    parser.add_argument("-i", "--numiterations", type=int, default=1, help="Number of iterations")
                    
    args = parser.parse_args()

    learner = LearningCorpus(args.freqcutoff, args.window, args.noise, args.numdim, args.inputfile)
    learner.train_on_corpus(args.numiterations, args.learningrate)
    print_vectors(learner.vocab, learner.wordVectors, args.outputfile)