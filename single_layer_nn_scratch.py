import numpy as np

class Neural_Network:
    def __init__(self,input_size, hidden_neurons, output_size, hidden_weights=None, output_weights=None):
        if hidden_weights is not None:
            self.hidden_weights = hidden_weights
        else:
            self.hidden_weights = np.random.rand(input_size, hidden_neurons)
        self.bias_hidden =  np.random.rand(1, hidden_neurons)
            
        if output_weights is not None:
            self.hidden_weights = output_weights
        else:
            self.output_weights = np.random.rand( hidden_neurons, output_size)
        self.bias_output =  np.random.rand(1, output_size)
            
        self.input_size = input_size
        self.hidden_neurons = hidden_neurons
        self. output_size =  output_size
        
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
        
    def forward(self,x):
        hidden_output =self.sigmoid(np.dot(x,self.hidden_weights)+ self.bias_hidden)
        output = self.sigmoid(np.dot( hidden_output, output_weights)+ self.bias_output)
        return hidden_output, output
    
input_size = int( input("input_size"))
hidden_neurons =int(input("hidden_neurons"))
output_size =  int(input("output_size")) 
input_data = np.array([float(input(f" enter input data {i+1}:")) for i in range(input_size)])
hidden_weights = np.array([[float(input(f" enter input data {i+1}- {j+1} hidden layer:")) for j in range(hidden_neurons)]for i in range(input_size)])
output_weights= np.array([[float(input(f" enter input data {i+1}- {j+1} output layer:")) for j  in range(output_size)]for i in range(hidden_neurons)])

network =Neural_Network(input_size, hidden_neurons, output_size, hidden_weights, output_weights) 
hidden_output, output = network.forward(input_data.reshape(1,-1))

print("weights in hidden layer", hidden_weights)
print("weights in output layer", output_weights)
print("Hidden output of neuron",hidden_output)
print("output of neuron",output)

            
            
        
