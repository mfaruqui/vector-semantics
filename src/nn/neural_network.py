import sys
import numpypy as np
import random
import math

N_POS = 1.2
N_NEG = 0.5
DEL_MAX = 50.0
DEL_MIN = 1e-6

def sign(val):
    
    if val < 0:
        return -1
    else:
        return 1

class NeuralNetwork:
    
    def __init__(self, nodesInNeuralLayers):
        
        self.nodesInNeuralLayers = list(nodesInNeuralLayers)
        self.numNeuralLayers = len(self.nodesInNeuralLayers)
        self.neuralLayers = []
        
        for i, lenLayer in enumerate(self.nodesInNeuralLayers):
            
            if lenLayer < 1:
                sys.stderr.write("Unallowable num node values, exiting...\n")
                sys.exit()
            
            if i < self.numNeuralLayers-2:
                self.neuralLayers.append( NeuralLayer(i, lenLayer, self.nodesInNeuralLayers[i+1]) )
            elif i == self.numNeuralLayers-2:
                #The final layer has no bias node
                self.neuralLayers.append( NeuralLayer(i, lenLayer, self.nodesInNeuralLayers[i+1]-1) )
            else:
                self.neuralLayers.append( NeuralLayer(i, lenLayer, None) )
        
        self.initialize_edges()
                
    def initialize_edges(self):
        
        for i in range(0, self.numNeuralLayers-1):
            for j in range(len(self.neuralLayers[i].transMat)):
                for k in range(len(self.neuralLayers[i].transMat[j])):
                    self.neuralLayers[i].transMat[j][k] = random.randint(-100,100)/100.
                    
                if i != self.numNeuralLayers-2:
                    self.neuralLayers[i].transMat[j][k] = 0.

    #Push the input value through all the layers of the network            
    def propagate_input(self, inputList):
        
        self.neuralLayers[0].reset_node_input_values(inputList)
        self.neuralLayers[0].reset_node_output_values(inputList)
        
        for neuralLayerNum, neuralLayer in enumerate(self.neuralLayers):
            if neuralLayerNum < self.numNeuralLayers-1:
                inputToNextLayer = neuralLayer.nodesOutput.dot(neuralLayer.transMat)
                self.neuralLayers[neuralLayerNum+1].compute_and_set_node_values(inputToNextLayer)
    
    #Follows the Gradient Descent algorithm given in Russell & Norvig            
    def train_network(self, exampleList, epoch, learningRate):
        
        for timeStep in range(epoch):
            for exampleNum, example in enumerate(exampleList):
                
                #Read the example and do the sanity checks
                inputList, outputList = example
                if len(inputList) != self.nodesInNeuralLayers[0] or len(outputList) != self.nodesInNeuralLayers[-1]:
                    sys.stderr.write("Incompatible input, exiting...\n")
                    sys.exit()
            
                #Propagate the input through the network
                #Add 1. for the bias node input   
                self.propagate_input(inputList+[1.])
                
                #Calculate error at the output layer
                self.neuralLayers[-1].calculate_and_set_delta_backprop(None, outputList)
                
                #Propagate the error back to all layers except for the input layer
                for i in range(self.numNeuralLayers-2, 0, -1):
                    self.neuralLayers[i].calculate_and_set_delta_backprop(self.neuralLayers[i+1])
                
                #Reset weights of all the edges of all layers
                for i in range(0, self.numNeuralLayers-1):
                    self.neuralLayers[i].reset_weights_backprop(learningRate, self.neuralLayers[i+1])
                
                '''
                #Run R-prop
                for i, neuralLayer in range(self.numNeuralLayers-2,0,-1):
                    neuralLayer.run_rprop(self.neuralLayers[i+1])
                '''
                
        for example in exampleList:
            i, o = example            
            self.propagate_input(i+[1.])
            print example, self.neuralLayers[-1].nodesOutput
                    
class NeuralLayer:
    
    def __init__(self, neuralLayerNum, numNodes, nextNeuralLayerNumNodes):
        
        if nextNeuralLayerNumNodes != None:
            self.numNodes = numNodes+1
            self.isOutputLayer = False
            self.transMat = np.zeros([self.numNodes, nextNeuralLayerNumNodes+1], dtype=float)
        else:
            #Output layer
            self.numNodes = numNodes
            self.isOutputLayer = True
            self.transMat = None
        
        self.neuralLayerNum = neuralLayerNum
        self.nodesInput = np.zeros([self.numNodes], dtype=float)
        self.nodesOutput = np.zeros([self.numNodes], dtype=float)
        self.nodesDelta = np.zeros([self.numNodes], dtype=float)
        
    def reset_node_input_values(self, inputValues):
        
        for i, value in enumerate(inputValues):
            self.nodesInput[i] = value
            
    def reset_node_output_values(self, outputValues):
        
        for i, value in enumerate(outputValues):
            self.nodesOutput[i] = value
        
        #If not output layer, reset value of bias node to 1.
        if not self.isOutputLayer:
            self.nodesOutput[i] = 1.0
    
    def compute_and_set_node_values(self, inputValues):
        
        self.reset_node_input_values(inputValues)
        
        values = []
        for val in inputValues:
            values.append(self.activation_function(val))
            
        self.reset_node_output_values(values)
        del values
        
    def activation_function(self, val):
        
        #Logit function
        return 1./(1+math.exp(-1*val))
        
    def activation_derivative(self, val):
        
        #Logit function derivative
        return math.exp(-1*val)/(1+math.exp(-1*val))**2
        
    def calculate_and_set_delta_backprop(self, nextNeuralLayer, outputList=None):
        
        if outputList != None:
            #sanity check that this is the output layer
            assert self.transMat == None
            for i, node in enumerate(self.nodesDelta):
                self.nodesDelta[i] = -1*self.activation_derivative(self.nodesInput[i])*(outputList[i]-self.nodesOutput[i])
        else:
            #calculating delta for an internal layer
            for i, node in enumerate(self.nodesDelta[:-1]):
                errorFromNextLayer = self.transMat[i].dot(nextNeuralLayer.nodesDelta)
                self.nodesDelta[i] = -1*self.activation_derivative(self.nodesInput[i])*errorFromNextLayer
                
    def reset_weights_backprop(self, learningRate, nextNeuralLayer):
        
        if not nextNeuralLayer.isOutputLayer:
            #Since the next layer has a bias node, exclude that from computation
            for i, weights in enumerate(self.transMat):
                self.transMat[i][:-1] -= learningRate * self.nodesOutput[i] * nextNeuralLayer.nodesDelta[:-1]
        else:
            #Final layer has no bias nodes
            for i, weights in enumerate(self.transMat):
                self.transMat[i] -= learningRate * self.nodesOutput[i] * nextNeuralLayer.nodesDelta
                
    def run_rprop(self, nextNeuralLayer):
        pass
        