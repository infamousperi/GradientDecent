from typing import Tuple
import numpy as np
from Classes.LinearLayer import LinearLayer


class NeuralNetwork:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, learning_rate: float = 0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Initialize layers with learning rate
        self.hidden_layer = LinearLayer(input_dim, hidden_dim, learning_rate)
        self.output_layer = LinearLayer(hidden_dim, output_dim, learning_rate)

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        # Hidden layer activation
        z_hidden = self.hidden_layer.forward_pass(input_data)
        a_hidden = stable_sigmoid(z_hidden)  # Use the stable version

        # Output layer activation
        z_output = self.output_layer.forward_pass(a_hidden)
        a_output = stable_sigmoid(z_output)  # Use the stable version

        return a_output

    def backward_pass(self, input_data: np.ndarray, output_gradient: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Forward pass within backward pass (to calculate activations)
        z_hidden = self.hidden_layer.forward_pass(input_data)
        a_hidden = 1 / (1 + np.exp(-z_hidden))

        z_output = self.output_layer.forward_pass(a_hidden)
        a_output = stable_sigmoid(z_output)

        # Output layer gradients (using calculated a_output)
        dA_output = output_gradient * (a_output * (1 - a_output))
        dZ_output = dA_output * (a_output * (1 - a_output))
        dWeights_output = np.dot(dZ_output.T, a_hidden)
        dBiases_output = np.sum(dZ_output, axis=0)
        #print("dWeights_output shape:", dWeights_output.shape)
        #print("Output layer weights shape:", self.output_layer.weights.shape)

        # Hidden layer gradients
        dA_hidden = np.dot(dZ_output, self.output_layer.weights)
        dZ_hidden = dA_hidden * (a_hidden * (1 - a_hidden))
        dWeights_hidden = np.dot(input_data.T, dZ_hidden).reshape(self.hidden_layer.weights.shape)
        dBiases_hidden = np.sum(dZ_hidden, axis=0)

        return dWeights_hidden, dBiases_hidden, dWeights_output, dBiases_output

    def parameter_update(self, dWeights_hidden: np.ndarray, dBiases_hidden: np.ndarray, dWeights_output: np.ndarray, dBiases_output: np.ndarray) -> None:
        # Update hidden layer parameters
        self.hidden_layer.parameter_update(dWeights_hidden, dBiases_hidden)
        # Update output layer parameters
        self.output_layer.parameter_update(dWeights_output, dBiases_output)


def stable_sigmoid(x):
    "Compute sigmoid function in a way that avoids overflow."
    positive_mask = (x >= 0)
    negative_mask = (x < 0)
    z = np.zeros_like(x)
    z[positive_mask] = np.exp(-x[positive_mask])
    z[negative_mask] = np.exp(x[negative_mask])
    top = np.ones_like(x)
    top[negative_mask] = z[negative_mask]
    return top / (1 + z)