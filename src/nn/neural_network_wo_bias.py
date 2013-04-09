import sys
import numpypy as np
import random
import math
import itertools

class NeuralNetwork:
    
    def __init__(self, nodesInNeuralLayers):
        
        self.nodesInNeuralLayers = list(nodesInNeuralLayers)
        self.numNeuralLayers = len(self.nodesInNeuralLayers)
        self.neuralLayers = []
        
        for i, lenLayer in enumerate(self.nodesInNeuralLayers):
            
            if lenLayer < 1:
                sys.stderr.write("Unallowable num node values, exiting...\n")
                sys.exit()
            
            if i < self.numNeuralLayers-1:
                self.neuralLayers.append( NeuralLayer(i, lenLayer, self.nodesInNeuralLayers[i+1]) )
            else:
                self.neuralLayers.append( NeuralLayer(i, lenLayer, None) )
        
    def randomly_initialize_edges(self):
        
        for i in range(0, self.numNeuralLayers-1):
            for j in range(len(self.neuralLayers[i].transMat)):
                for k in range(len(self.neuralLayers[i].transMat[j])):
                    self.neuralLayers[i].transMat[j][k] = random.randint(0,100)/100.
                    
            #print self.neuralLayers[i].transMat
                    
    #Push the input value through all the layers of the network            
    def propagate_input(self, inputList):
        
        self.neuralLayers[0].nodesInput = np.array(inputList)
        self.neuralLayers[0].nodesOutput = np.array(inputList)
        
        for neuralLayerNum, neuralLayer in enumerate(self.neuralLayers[:-1]):
            inputToNextLayer = neuralLayer.nodesOutput.dot(neuralLayer.transMat)
            self.neuralLayers[neuralLayerNum+1].calculate_and_set_node_values(inputToNextLayer)
    
    #Follows the Gradient Descent algorithm given in Russell & Norvig            
    def train_network(self, exampleList, epoch, learningRate):
        
        self.randomly_initialize_edges()
        
        for timeStep in range(epoch):
            for exampleNum, example in enumerate(exampleList):
                
                #Read the example and do the sanity checks
                if timeStep == 0:
                    self.input_sanity_check(example)
                
                inputList, outputList = example
                #Propagate the input through the network
                self.propagate_input(inputList)
                
                #print self.neuralLayers[-1].nodesOutput
                
                #Calculate error at the output layer
                self.neuralLayers[-1].calculate_and_set_delta_backprop(None, outputList)
                
                #Propagate the error back to all layers except for the input layer
                for i in range(self.numNeuralLayers-2, 0, -1):
                    self.neuralLayers[i].calculate_and_set_delta_backprop(self.neuralLayers[i+1])
                
                #Reset weights of all the edges of all layers
                for i in range(0, self.numNeuralLayers-1):
                    self.neuralLayers[i].reset_weights_backprop(learningRate, self.neuralLayers[i+1])

        self.print_output(exampleList)
                    
    def print_output(self, exampleList):
        
        for layer in self.neuralLayers:
            print layer.transMat
        
        for example in exampleList:
            i, o = example            
            self.propagate_input(i)
            print example, self.neuralLayers[-1].nodesOutput
            
    def input_sanity_check(self, example):
        
        inputList, outputList = example
        if len(inputList) != self.nodesInNeuralLayers[0] or len(outputList) != self.nodesInNeuralLayers[-1]:
            sys.stderr.write("Incompatible input, exiting...\n")
            sys.exit()
        
class NeuralLayer:
    
    def __init__(self, neuralLayerNum, numNodes, nextNeuralLayerNumNodes):
        
        self.numNodes = numNodes
        self.neuralLayerNum = neuralLayerNum
        self.nodesInput = np.zeros([self.numNodes], dtype=float)
        self.nodesOutput = np.zeros([self.numNodes], dtype=float)
        self.nodesDelta = np.zeros([self.numNodes], dtype=float)
        
        if nextNeuralLayerNumNodes != None:
            self.isOutputLayer = False
            self.transMat = np.ones([self.numNodes, nextNeuralLayerNumNodes], dtype=float)
        else:
            self.isOutputLayer = True
            self.transMat = None
        
    def calculate_and_set_node_values(self, inputValues):
        
        self.nodesInput = np.array(inputValues)
        self.nodesOutput = np.array([self.activation_function(val) for val in inputValues])
        
    def activation_function(self, val):
        
        #Logit function
        return 1./(1+math.exp(-1*val))
        
        #tanh function
        #return (math.exp(val)-math.exp(-1*val))/(math.exp(val)+math.exp(-1*val))
        
    def activation_derivative(self, val):
        
        #Logit function derivative
        return math.exp(-1*val)/(1+math.exp(-1*val))**2
        
        #Tanh function derivative
        #return 1-self.activation_function(val)**2
        
    def calculate_and_set_delta_backprop(self, nextNeuralLayer, outputList=None):
        
        if outputList != None and nextNeuralLayer == None:
            for i, node in enumerate(self.nodesDelta):
                self.nodesDelta[i] = self.activation_derivative(self.nodesInput[i])*(outputList[i]-self.nodesOutput[i])
        else:
            #calculating delta for an internal layer
            for i, node in enumerate(self.nodesDelta):
                errorFromNextLayer = self.transMat[i].dot(nextNeuralLayer.nodesDelta)
                self.nodesDelta[i] = self.activation_derivative(self.nodesInput[i])*errorFromNextLayer
                
    def reset_weights_backprop(self, learningRate, nextNeuralLayer):
        
        for i in range(len(self.transMat)):
            self.transMat[i] -= -1 *learningRate * self.nodesOutput[i] * nextNeuralLayer.nodesDelta