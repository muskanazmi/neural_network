import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert target labels to binary values (0 for benign, 1 for malignant) for training, validation, and test sets
y_train_binary = np.where(y_train == 1, 1, 0)
y_val_binary = np.where(y_val == 1, 1, 0)
y_test_binary = np.where(y_test == 1, 1, 0)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, l1_penalty):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.l1_penalty = l1_penalty  # Regularization parameter
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.biases_input_hidden = np.zeros((1, self.hidden_size))
        
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.biases_hidden_output = np.zeros((1, self.output_size))
        
        # Initialize lists to store training and validation losses
        self.train_losses = []
        self.val_losses = []
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Input to hidden layer
        self.hidden_output = self.sigmoid(np.dot(X, self.weights_input_hidden) + self.biases_input_hidden)
        
        # Hidden to output layer
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.biases_hidden_output)
        return self.output
    
    def backward(self, X, y, learning_rate):
        # Calculate loss
        loss = y - self.output
        
        # Compute gradients for output layer
        output_delta = loss * self.sigmoid_derivative(self.output)
        
        # Compute gradients for hidden layer
        hidden_loss = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_loss * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases with L1 regularization
        self.weights_hidden_output += (np.dot(self.hidden_output.T, output_delta) - self.l1_penalty * np.sign(self.weights_hidden_output)) * learning_rate
        self.biases_hidden_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        
        self.weights_input_hidden += (np.dot(X.T, hidden_delta) - self.l1_penalty * np.sign(self.weights_input_hidden)) * learning_rate
        self.biases_input_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
    
    def train(self, X_train, y_train, X_val, y_val, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass for training set
            output_train = self.forward(X_train)
            
            # Backward pass for training set
            self.backward(X_train, y_train, learning_rate)
            
            # Calculate training loss
            train_loss = np.mean(np.square(y_train - output_train))
            self.train_losses.append(train_loss)
            
            # Forward pass for validation set
            output_val = self.forward(X_val)
            
            # Calculate validation loss
            val_loss = np.mean(np.square(y_val - output_val))
            self.val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        # Plotting
        epochs_range = range(epochs)
        plt.plot(epochs_range, self.train_losses, label='Train Loss')
        plt.plot(epochs_range, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.legend()
        plt.show()
    
    def predict(self, X):
        return np.round(self.forward(X))

# Define neural network parameters
input_size = X_train.shape[1]
hidden_size = 8  
output_size = 1  
l1_penalty = 0.1  # Adjust this parameter for the desired strength of regularization

# Initialize the neural network with L1 regularization
nn = NeuralNetwork(input_size, hidden_size, output_size, l1_penalty)

# Train the neural network
nn.train(X_train, y_train_binary.reshape(-1, 1), X_val, y_val_binary.reshape(-1, 1), epochs=100, learning_rate=0.1)

# Predict on training set
y_train_pred = nn.predict(X_train)

# Calculate training accuracy
training_accuracy = np.mean(y_train_pred == y_train_binary.reshape(-1, 1)) * 100
print(f"Training Accuracy: {training_accuracy:.2f}%")

# Predict on test set
y_test_pred = nn.predict(X_test)

# Calculate test accuracy
test_accuracy = np.mean(y_test_pred == y_test_binary.reshape(-1, 1)) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Predict on validation set
y_val_pred = nn.predict(X_val)

# Calculate validation accuracy
val_accuracy = np.mean(y_val_pred == y_val_binary.reshape(-1, 1)) * 100
print(f"Validation Accuracy: {val_accuracy:.2f}%")
