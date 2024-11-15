#Layer and Inputlayer classes
# Layers represent the neural network's structure
# Each layer should store its weights, biases, and activation function

# Input Layer, this layer holds the input value and passes them to the first hidden layer.
import numpy as np
from Activations import ReLU

class InputLayer:
    def _init_(self,width):
        self.output = np.zeros(width)

    def setOutput(self, values):
        self.output = values

    def getNodes(self):
        return self.output

#Layer Class manages forward propagation, weights, biases, and activation for hidden/output layers.
class Layer:
    def _init_(self, width, biases, activation, layer_index, GPU=False):
        self.width = width
        self.biases = np.array(biases)
        self.activation = activation
        self.layer_index = layer_index
        self.output = np.zeros(width)
        self.weights = None

    def connectPreviousLayer(self, prev_layer, weights):
        #Connect to the previous layer with weights.
        self.weights = weights
        self.prev_layer = prev_layer

    def forward(self):
        # z (pre-activation) is weighted sum of inputs plus bias for each neuron. z = input*weights + bias
        z = np.dot(self.prev_layer.output, self.weights) + self.biases
        # output = activation(z)
        self.output = self.activation(z)
        return self.output