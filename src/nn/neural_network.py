import sys
import numpypy as np
import random
import math
import itertools

N_POS = 1.2
N_NEG = 0.5
DEL_INIT = 0.1
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
        
    def set_bias_node_incoming_edges_zero(self):
        
        for i in range(0, self.numNeuralLayers-1):
            for j in range(len(self.neuralLayers[i].transMat)):
                if i != self.numNeuralLayers-2:
                    self.neuralLayers[i].transMat[j][-1] = 0.
                
    def randomly_initialize_edges(self):
        
        for i in range(0, self.numNeuralLayers-1):
            for j in range(len(self.neuralLayers[i].transMat)):
                for k in range(len(self.neuralLayers[i].transMat[j])):
                    self.neuralLayers[i].transMat[j][k] = random.randint(-100,100)/100.
                    
                if i != self.numNeuralLayers-2:
                    self.neuralLayers[i].transMat[j][k] = 0.

    #Push the input value through all the layers of the network            
    def propagate_input(self, inputList):
        
        self.neuralLayers[0].nodesInput = np.array(inputList)
        self.neuralLayers[0].reset_node_output_values(inputList)
        
        for neuralLayerNum, neuralLayer in enumerate(self.neuralLayers):
            if neuralLayerNum < self.numNeuralLayers-1:
                inputToNextLayer = neuralLayer.nodesOutput.dot(neuralLayer.transMat)
                self.neuralLayers[neuralLayerNum+1].compute_and_set_node_values(inputToNextLayer)
    
    #Follows the Gradient Descent algorithm given in Russell & Norvig            
    def train_backprop(self, exampleList, epoch, learningRate):
        
        self.randomly_initialize_edges()
        self.set_bias_node_incoming_edges_zero()
        
        for timeStep in range(epoch):
            for exampleNum, example in enumerate(exampleList):
                
                #Read the example and do the sanity checks
                self.input_sanity_check(example)
                
                #Propagate the input through the network
                #Add 1. for the bias node input   
                inputList, outputList = example
                self.propagate_input(inputList+[1.])
                
                #Calculate error at the output layer
                self.neuralLayers[-1].calculate_and_set_delta_backprop(None, outputList)
                
                #Propagate the error back to all layers except for the input layer
                for i in range(self.numNeuralLayers-2, 0, -1):
                    self.neuralLayers[i].calculate_and_set_delta_backprop(self.neuralLayers[i+1])
                
                #Reset weights of all the edges of all layers
                for i in range(0, self.numNeuralLayers-1):
                    self.neuralLayers[i].reset_weights_backprop(learningRate, self.neuralLayers[i+1])
                
        self.print_output(exampleList)
                    
    def train_rprop(self, exampleList, epoch):
        
        for timeStep in range(epoch):
            for exampleNum, example in enumerate(exampleList):
                
                inputList, outputList = example
                self.input_sanity_check(inputList, outputList)
            
                self.propagate_input(inputList+[1.])
                self.neuralLayers[-1].run_rprop(None, outputList)
                
                for i in range(self.numNeuralLayers-2, 0, -1):
                    self.neuralLayers[i].run_rprop(self.neuralLayers[i+1])
                    
        self.print_output(exampleList)
            
    def print_output(self, exampleList):
        
        for layer in self.neuralLayers:
            print layer.transMat
                
        for example in exampleList:
            i, o = example            
            self.propagate_input(i+[1.])
            print example, self.neuralLayers[-1].nodesOutput
            
    def input_sanity_check(self, example):
        
        inputList, outputList = example
        if len(inputList) != self.nodesInNeuralLayers[0] or len(outputList) != self.nodesInNeuralLayers[-1]:
            sys.stderr.write("Incompatible input, exiting...\n")
            sys.exit()
        
class NeuralLayer:
    
    def __init__(self, neuralLayerNum, numNodes, nextNeuralLayerNumNodes):
        
        if nextNeuralLayerNumNodes != None:
            self.numNodes = numNodes+1
            self.isOutputLayer = False
            self.transMat = np.ones([self.numNodes, nextNeuralLayerNumNodes+1], dtype=float)
            self.rpropDelta = DEL_INIT*np.ones([self.numNodes, nextNeuralLayerNumNodes+1], dtype=float)
            self.rpropDelEDelW = np.ones([self.numNodes, nextNeuralLayerNumNodes+1], dtype=float)
            self.rpropEdgesDelta = np.ones([self.numNodes, nextNeuralLayerNumNodes+1], dtype=float)
        else:
            #Output layer
            self.numNodes = numNodes
            self.isOutputLayer = True
            self.transMat = None
            self.rpropDelta = None
            self.rpropDelEDelW = None
            self.rpropEdgesDelta = None
        
        self.neuralLayerNum = neuralLayerNum
        self.nodesInput = np.zeros([self.numNodes], dtype=float)
        self.nodesOutput = np.zeros([self.numNodes], dtype=float)
        self.nodesDelta = np.zeros([self.numNodes], dtype=float)
        
        #set the bias node output to 1. and it never changes
        if not self.isOutputLayer:
            self.nodesOutput[-1] = 1.
        
    def reset_node_output_values(self, outputValues):
        
        if self.isOutputLayer:
            self.nodesOutput = np.array(outputValues)
        else:
            #Dont change the bias node value ever
            self.nodesOutput[:-1] = np.array(outputValues[:-1])
    
    def compute_and_set_node_values(self, inputValues):
        
        #If not output layer, the input to the bias node should be zero
        if not self.isOutputLayer:
            assert inputValues[-1] == 0.0
         
        self.nodesInput = np.array(inputValues)   
        self.reset_node_output_values([self.activation_function(val) for val in inputValues])
        
    def activation_function(self, val):
        
        #Logit function
        return 1./(1+math.exp(-1*val))
        
    def activation_derivative(self, val):
        
        #Logit function derivative
        return math.exp(-1*val)/(1+math.exp(-1*val))**2
        
    def calculate_and_set_delta_backprop(self, nextNeuralLayer, outputList=None):
        
        if outputList != None and nextNeuralLayer == None:
            for i, node in enumerate(self.nodesDelta):
                self.nodesDelta[i] = self.activation_derivative(self.nodesInput[i])*(outputList[i]-self.nodesOutput[i])
        else:
            #calculating delta for an internal layer
            for i, node in enumerate(self.nodesDelta[:-1]):
                errorFromNextLayer = self.transMat[i].dot(nextNeuralLayer.nodesDelta)
                self.nodesDelta[i] = self.activation_derivative(self.nodesInput[i])*errorFromNextLayer
                
    def reset_weights_backprop(self, learningRate, nextNeuralLayer):
        
        if not nextNeuralLayer.isOutputLayer:
            #Since the next layer has a bias node, exclude that from computation
            for i, weights in enumerate(self.transMat):
                self.transMat[i][:-1] -= -1*learningRate * self.nodesOutput[i] * nextNeuralLayer.nodesDelta[:-1]
        else:
            #Final layer has no bias nodes
            for i, weights in enumerate(self.transMat):
                self.transMat[i] -= -1*learningRate * self.nodesOutput[i] * nextNeuralLayer.nodesDelta
                
    def run_rprop(self, nextNeuralLayer, outputList=None):
        
        if outputList != None:
            
            for i, node in enumerate(self.nodesDelta):
                self.nodesDelta[i] = -1*self.activation_derivative(self.nodesInput[i])*(outputList[i]-self.nodesOutput[i])
        else:
            
            for i, node in enumerate(self.nodesDelta):
                
                errorFromNextLayer = self.transMat[i].dot(nextNeuralLayer.nodesDelta)
                self.nodesDelta[i] = -1*self.activation_derivative(self.nodesInput[i])*errorFromNextLayer
                
                if not nextNeuralLayer.isOutputLayer:
                    iterTransMat = enumerate( itertools.islice(self.transMat[i], 0, len(self.transMat[i])-1 ))
                else:
                    iterTransMat = enumerate( itertools.islice(self.transMat[i], 0, len(self.transMat[i]) ))
                    
                for j, edgeWeight in iterTransMat:
                    newDelEDelW = -1*self.nodesOutput[i] * nextNeuralLayer.nodesDelta[j]
                    
                    if self.rpropDelEDelW[i][j] * newDelEDelW > 0:
                        self.rpropDelta[i][j] = min(N_POS*self.rpropDelta[i][j], DEL_MAX)
                        self.rpropEdgesDelta[i][j] = -1*sign(newDelEDelW) * self.rpropDelta[i][j]
                        self.transMat[i][j] = edgeWeight + self.rpropEdgesDelta[i][j]
                        self.rpropDelEDelW[i][j] = newDelEDelW
                        
                    elif self.rpropDelEDelW[i][j] * newDelEDelW < 0:
                        self.rpropDelta[i][j] = max(N_NEG*self.rpropDelta[i][j], DEL_MIN)
                        #rpropEdgesDelta remains the same
                        self.transMat[i][j] = edgeWeight - self.rpropEdgesDelta[i][j]
                        self.rpropDelEDelW[i][j] = 0.
                        
                    else:
                        #rpropDelta remains the same as last time
                        self.rpropEdgesDelta[i][j] = -1*sign(newDelEDelW) * self.rpropDelta[i][j]
                        self.transMat[i][j] = edgeWeight + self.rpropEdgesDelta[i][j]
                        self.rpropDelEDelW[i][j] = newDelEDelW