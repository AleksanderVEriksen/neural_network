
import numpy as np
# single neuron: output = sum(inputs * weights) + bias
# output = activation(output)
print("----- Single neuron -----")
inputs = [1.0,2.0,3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = (inputs[0]*weights[0] +
          inputs[1]*weights[1] +
          inputs[2]*weights[2] +
          inputs[3]*weights[3] + bias
          )
print(output)
# With dot product
print("Applied dot product:") 
outputs = np.dot(weights, inputs) + bias
print(outputs)

# Multiple neurons
print("----- Multiple neurons -----")
inputs = [1.0,2.0,3.0, 2.5]

weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

output = [
        # Neuron 1:
        inputs[0]*weights1[0] +
        inputs[1]*weights1[1] +
        inputs[2]*weights1[2] +
        inputs[3]*weights1[3] + bias1,
        
          

        # Neuron 2:
        inputs[0]*weights2[0] +
        inputs[1]*weights2[1] +
        inputs[2]*weights2[2] +
        inputs[3]*weights2[3] + bias2,
          
          

        # Neuron 3:
        inputs[0]*weights3[0] +
        inputs[1]*weights3[1] +
        inputs[2]*weights3[2] +
        inputs[3]*weights3[3] + bias3  
          ]
print(output)

# With dot product
print("Applied dot product:") 
outputs =[np.dot(weights1, inputs) + bias1,
          np.dot(weights2, inputs) + bias2,
          np.dot(weights3, inputs) + bias3,
          ]
print(outputs)


# Optimized code
print("----- Optimized code for multiple neurons -----")
inputs = [1,2,3,2.5]

weights = [[0.2, 0.8, -0.5, 1],
[0.5, -0.91, 0.26, -0.5],
[-0.26, -0.27, 0.17, 0.87]]

biases = [2,3,0.5]

layer_outputs = []

for neuron_weights, neuron_biases in zip(weights, biases): # Iterate through both weight and bias list
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights): # Iterate through both inputs and neuron_weights list
        neuron_output += n_input * weight
    neuron_output+=neuron_biases
    layer_outputs.append(neuron_output)
print(layer_outputs)
# With dot product 
print("Applied dot product:")
layer_outputs = np.dot(weights, inputs) + biases
print(layer_outputs)


#### Dot products
print("----- Dot product -----")
a = [1,2,3]
b = [2,3,4]

dot_product = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
print(dot_product)

# Matrix dot product and transpose

a = [1,2,3]
b = [2,3,4]
a = np.array([a])
b = np.array([b]).T
np.dot(a,b)

# Applying logic to 1-layer network
print("-- Matrix dot product --")
inputs = [[1.0,2.0,3.0,2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]
          ]

weights = [[0.2, 0.8, -0.5, 1.0],
[0.5, -0.91, 0.26, -0.5],
[-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

outputs = np.dot(inputs, np.array(weights).T) + biases  # swapped inputs and weights, therefor transpose
                                                        # Result array will be sample related rather than neuron related

print(outputs)


