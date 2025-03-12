import numpy as np
import math
# Multiple layers
inputs = [[1.0,2.0,3.0,2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]
          ]

weights = [[0.2, 0.8, -0.5, 1],
[0.5, -0.91, 0.26, -0.5],
[-0.26, -0.27, 0.17, 0.87]]

biases = [2,3,0.5]

weights2 = [[0.1, -0.14, 0.5],
[-0.5, 0.12, -0.33],
[-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print("-- Two layer network --")
print(layer2_outputs)
# Non linear data

from nnfs.datasets import spiral_data
import nnfs
import matplotlib.pyplot as plt
nnfs.init()

X,y = spiral_data(samples=100, classes=3)
plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
plt.show()


# Class initialization of a Dense layer

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # Adds weights with random values between n_inputs and n_neurons
        self.biases = np.zeros((1, n_neurons)) # Adds biases equal to neurons valued to 0
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# Testing layers
print("-- Testing layers with spiral data --")
dense1 = Layer_Dense(2,3)
dense1.forward(X)
print(dense1.output[:5])

# ReLU Activation Function Code
print("-- ReLU --")
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

output = []
for i in inputs:
    if i>0:
        output.append(i)
    else:
        output.append(0)
print(output)
# Better
output = []
for i in inputs:
    output.append(max(0,i))
print(output)
# Optimal
output = []
output = np.maximum(0, inputs)
print(output)

# Creating ReLU class

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
# Testing with ReLU
print("-- ReLU added to Layer --")
activation1 = Activation_ReLU()
activation1.forward(dense1.output)
print(activation1.output[:5])
# Softmax scratch
layer_outputs=[4.8, 1.21, 2.385]
E = math.e
exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output)
print('exponentiated values: ')
print(exp_values)

norm_base = sum(exp_values)
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)
print('Normalized exponentiated values: ')
print(norm_values)
print('Sum of normalized values: ',sum(norm_values))

# Softmax numpy
inputs = [[0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100],
          [1, 3, -2, 5.5, -4.2, 2.2, 6.7, -200]
          ]
layer_outputs=[4.8, 1.21, 2.385]
exp_values = np.exp(layer1_outputs)
norm_values = exp_values / np.sum(exp_values)
# For batches
exp_values = np.exp(inputs)
probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Creation of Class Softmax

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
# Apply to test of the network
# Create dataset
X, y=spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1=Layer_Dense(2,3)
# Create ReLU activation (to be used with Dense layer):
activation1=Activation_ReLU()
# Create second Dense layer with 3 input features (as we take output# of previous layer here) and 3 output values
dense2=Layer_Dense(3,3)
# Create Softmax activation (to be used with Dense layer):
activation2=Activation_Softmax()
# Make a forward pass of our training data through this layer
dense1.forward(X)
# Make a forward pass through activation function# it takes the output of first dense layer here
activation1.forward(dense1.output)
# Make a forward pass through second Dense layer
# # it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Make a forward pass through activation function
# # it takes the output of second dense layer here
activation2.forward(dense2.output)
# Let's see output of the first few samples:
print(activation2.output[:5])