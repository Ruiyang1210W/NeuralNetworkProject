# The model class ties the layers together to form a full neural network.
# It handles initialization, forward propagation, and connecting layers.
from Layers import Layer, InputLayer
import numpy as np
from Activations import ReLU, Activation

class Model:
    #Connect each layer to the next by setting weights and biases
    def _init_(self, width, depth, startweights=None, GPU=False):
        self.layers = [InputLayer(width)]
        for i in range(depth - 2):
            self.layers.append(Layer(width, [0] * width, ReLU(), i+1, GPU))
        self.layers.append(Layer(width,[0] * width, Acitvation(), depth - 1, GPU))

        for i, layer in enumerate(self.layers):
            if i > 0:
                weights = startweights[i-1] if startweights else np.random.randn(width, width)
                layer.connectPreviousLayer(self.layers[i-1], weights)
            if i < len(self.layers) - 1:
                layer.connectNextLayer(self.layers[i+1])

    def forward(self, values):
        #forward pass
        self.layers[0].setOutput(values) #Starts at the input, each layer's output is calculated and passed to the next layer
        for layer in self.layers[1:]:
            layer.forward
        
    def backward(self, ytrue): # yture is the true labels
        # Use to compute gradients for each layer
        # start from the output layer and move backward, applying chain rule to calculate gradients
        # activation.derivative() is on each layer to calculate each gradient based on the error at each node
        self.layers[-1].backward(yture)
        for layer in reversed(self.layers[1:-1]):
            layer.backward()

    def update(self, learning_rate = 0.01):
        for layer in self.layers[1:]:
            layer.updateWeights(learning_rate)
