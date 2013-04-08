from neural_network import *

if __name__=='__main__':
    a = NeuralNetwork([2,4,2])
    #a.train_network([([1,1],[0]), ([0,0],[0]),([1,0],[1]),([0,1],[1])], 10000, 0.5)
    a.train_network([([1,0],[1,0]), ([2,0],[2,0])], 100000, 0.1)