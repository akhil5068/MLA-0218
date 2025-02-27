import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize the neural network
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.5):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # Initialize weights with small random values
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.input_nodes, self.hidden_nodes))
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.hidden_nodes, self.output_nodes))

        # Initialize biases
        self.bias_hidden = np.random.uniform(-1, 1, (1, self.hidden_nodes))
        self.bias_output = np.random.uniform(-1, 1, (1, self.output_nodes))

    # Forward Propagation
    def forward_propagation(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_layer_input)
        
        return self.output

    # Backpropagation and weight update
    def backward_propagation(self, X, y):
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    # Train the network
    def train(self, X, y, epochs=5000):
        for epoch in range(epochs):
            self.forward_propagation(X)
            self.backward_propagation(X, y)
            
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - self.output))
                print(f"Epoch {epoch}, Loss: {loss:.5f}")

    # Predict on new input data
    def predict(self, X):
        return self.forward_propagation(X)

# Create dataset (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([[0], [1], [1], [0]])  # Outputs

# Define the neural network
input_nodes = 2
hidden_nodes = 4
output_nodes = 1
learning_rate = 0.5

# Train the neural network
nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
nn.train(X, y, epochs=5000)

# Test the trained ANN
print("\nPredictions on XOR dataset:")
for sample in X:
    print(f"Input: {sample}, Predicted Output: {nn.predict(sample)}")
