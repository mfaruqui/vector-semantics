from neural_network_wo_bias import *

if __name__=='__main__':
    
    a = NeuralNetwork([2,2,1])
    a.train_network([([1,1],[0]), ([0,0],[0]),([1,0],[1]),([0,1],[1])], 1000, 0.1)
    #a.train_backprop([([1,100],[1,100]), ([0,0],[0,0]),([1,0],[1,0]),([0,1],[0,1])], 1000, 0.1)
    #a.train_rprop([([1,1],[0]), ([0,0],[0]),([1,0],[1]),([0,1],[1])], 1000)
    #a.train_rprop([([1,10],[1,10]), ([2,0],[2,0])], 1000)
    #a.train_network([([2,0,5],[2,0,5])], 10000, 0.1)