import numpy as np 


class Activation:
    def activate(self, input):
        #Base activation function
        return 1/(1 + np.exp(-input))

    def derivative(self, input):
        # Derivative of sigmoid for backprop
        return self(input) * (1-self(input))

class ReLU:
    def activate(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return np.where(x > 0, 1, 0)
