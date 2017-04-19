import numpy as np
import random

class ANN(object):
    def __init__(self, num_inputs, num_hidden_nodes, num_outputs, weights):
        self.weights = weights # this is an individual which is actually a list of weights (genome)

        self.num_hidden_nodes = num_hidden_nodes
        num_hidden_weights = (num_inputs+1) * num_hidden_nodes

        self.hidden_weights = np.array(weights[:num_hidden_weights]).reshape(num_hidden_nodes,num_inputs+1)
        self.output_weights = np.array(weights[num_hidden_weights:]).reshape(num_outputs,num_hidden_nodes+1)

    def activation(self,x):
        # x is the net input to the neuron (previously represented as "z" during the class)
        # a is the activation value ( a = activation(z) )
        # activation function could be sigmoid function: 1/(1+exp(-x))
        a = (1/(1+np.exp(-x)))
        return a
        
    def netInputsHiddenLayer(self, inputs):
        nodeSum = 0
        hiddenLayerSums = []
        size = 0
        for nodeCount in range (0, len(self.hidden_weights)):
            size = len(self.hidden_weights[nodeCount])
            inputCount = 0
            for weightCount in range(0, size):
                nodeSum += inputs[inputCount]*self.hidden_weights[nodeCount][weightCount]
                inputCount+=1
                if(inputCount > 784):       
                    inputCount = 0
                    hiddenLayerSums.append(nodeSum)
                    nodeSum = 0
        return hiddenLayerSums
                
    def activationsHiddenLayer(self, hiddenLayerSums):
        hiddenLayerNodes = []
        for num in range (0, len(hiddenLayerSums)):
            hiddenLayerNodes.append(self.activation(hiddenLayerSums[num]))
        return hiddenLayerNodes
        
    def netInputsOutputLayer(self, hiddenLayerNodes):
        outputLayerSums = [] 
        outputSum = 0
        inputCount = 0
        for nodeCount in range(0, len(self.output_weights)):
            size = len(self.output_weights[nodeCount])
            inputCount = 0
            for weightCount in range(0, size):
                outputSum += hiddenLayerNodes[inputCount]*self.output_weights[nodeCount][weightCount]
                inputCount += 1
                if(inputCount > 10):  
                    inputCount = 0
                    outputLayerSums.append(outputSum)
                    outputSum = 0
        return outputLayerSums
        
    def activationsOutputLayer(self, outputSums):
        outputNodes = []
        for num in range(0, 10):
            outputNodes.append(self.activation(outputSums[num]))
        return outputNodes

    def evaluate(self,inputs):
        hiddenLayerInputs = [1]
        for num in range(0, len(inputs)):
            hiddenLayerInputs.append(inputs[num])
        hiddenLayerSums = self.netInputsHiddenLayer(hiddenLayerInputs)
        hiddenLayerNodes = self.activationsHiddenLayer(hiddenLayerSums)
        #print(hiddenLayerNodes)
        hiddenLayerNodesWithBias = [1]
        for num in range(0, len(hiddenLayerNodes)):
            hiddenLayerNodesWithBias.append(hiddenLayerNodes[num])
        outputSums = self.netInputsOutputLayer(hiddenLayerNodesWithBias)
        outputs = self.activationsOutputLayer(outputSums)
        # Compute outputs from the fully connected feed-forward ANN:
        return outputs
    
