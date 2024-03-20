import numpy as np


class LinearLayer:
    # Initializes the linear layer
    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.zeros(output_size)

    # Forward pass of the linear layer
    def forward_pass(self, input_data):
        output_data = np.matmul(input_data, self.weights.T) + self.bias
        return output_data

    # Backward pass of the linear layer
    def backward_pass(self, input_data, output_gradient):
        input_gradient = np.matmul(output_gradient, self.weights)
        weight_gradient = np.matmul(output_gradient.T, input_data)
        bias_gradient = np.sum(output_gradient, axis=0)
        return input_gradient, weight_gradient, bias_gradient

    # Updates the weights and biases using gradient descent
    def parameter_update(self, weight_gradient, bias_gradient):
        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient
