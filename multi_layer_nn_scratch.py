import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights(input_size, hidden1_size, hidden2_size, output_size):
    np.random.seed(42)
    weights = {
        'hidden1': np.random.rand(input_size, hidden1_size),
        'hidden2': np.random.rand(hidden1_size, hidden2_size),
        'output': np.random.rand(hidden2_size, output_size)
    }
    return weights

def forward_propagation(inputs, weights):
    hidden1 = sigmoid(np.dot(inputs, weights['hidden1']))
    hidden2 = sigmoid(np.dot(hidden1, weights['hidden2']))
    output = sigmoid(np.dot(hidden2, weights['output']))
    return hidden1, hidden2, output

def mean_squared_error(targets, predictions):
    return np.mean((targets - predictions) ** 2)

def backpropagation(inputs, targets, weights, learning_rate):
    hidden1, hidden2, output = forward_propagation(inputs, weights)

    output_error = targets - output
    output_delta = output_error * sigmoid_derivative(output)

    hidden2_error = output_delta.dot(weights['output'].T)
    hidden2_delta = hidden2_error * sigmoid_derivative(hidden2)

    hidden1_error = hidden2_delta.dot(weights['hidden2'].T)
    hidden1_delta = hidden1_error * sigmoid_derivative(hidden1)

    weights['output'] += hidden2.T.dot(output_delta) * learning_rate
    weights['hidden2'] += hidden1.T.dot(hidden2_delta) * learning_rate
    weights['hidden1'] += inputs.T.dot(hidden1_delta) * learning_rate

    return weights

# User input for network architecture
input_neurons = int(input("Enter the number of input neurons: "))
hidden1_neurons = int(input("Enter the number of neurons in hidden layer 1: "))
hidden2_neurons = int(input("Enter the number of neurons in hidden layer 2: "))
output_neurons = 2  # Fixed 2 output neurons

# User input for target values
#targets = np.array([[float(input(f"Enter target value for sample {i + 1}: "))] for i in range(input_neurons)])
# User input for target values
targets = np.array([[float(input(f"Enter target value for sample {i + 1}: "))] for i in range(input_neurons)])


inputs = np.random.rand(input_neurons, input_neurons)  

weights = initialize_weights(input_neurons, hidden1_neurons, hidden2_neurons, output_neurons)

# Training loop
epochs = 2
learning_rate = 0.1

for epoch in range(epochs):
    for i in range(len(inputs)):
        input_sample = inputs[i:i+1]
        target_sample = targets[i:i+1]

        if epoch == 0 and i == 0:
            _, _, predictions = forward_propagation(inputs, weights)
            mse_before_backpropagation = mean_squared_error(targets, predictions)
            print(f'Mean Squared Error before the first backward propagation: {mse_before_backpropagation:.4f}')

        weights = backpropagation(input_sample, target_sample, weights, learning_rate)

        if epoch == 1 and i == 0:
            _, _, predictions = forward_propagation(inputs, weights)
            mse_before_second_backpropagation = mean_squared_error(targets, predictions)
            print(f'Mean Squared Error before the second backward propagation: {mse_before_second_backpropagation:.4f}')

_, _, predictions = forward_propagation(inputs, weights)
mse = mean_squared_error(targets, predictions)
print(f'\nFinal Predicted outcomes after the second iteration:\n{predictions}')
print(f'Final Mean Squared Error: {mse:.4f}')

# Print final weights
print("\nFinal Weights:")
print("Hidden Layer 1 Weights:\n", weights['hidden1'])
print("Hidden Layer 2 Weights:\n", weights['hidden2'])
print("Output Layer Weights:\n", weights['output'])
