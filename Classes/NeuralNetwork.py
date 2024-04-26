from typing import Tuple, List
import numpy as np
from Classes.LinearLayer import LinearLayer


class NeuralNetwork:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, hidden_layers: int,
                 learning_rate: float = 0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.hidden_layers = [LinearLayer(hidden_dim, hidden_dim, learning_rate) for _ in range(hidden_layers)]
        self.hidden_layers[0] = LinearLayer(input_dim, hidden_dim, learning_rate)  # First layer's input dimension
        self.output_layer = LinearLayer(hidden_dim, output_dim, learning_rate)

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        current_input = input_data
        for layer in self.hidden_layers:
            z = layer.forward_pass(current_input)
            current_input = stable_sigmoid(z)

        z_output = self.output_layer.forward_pass(current_input)
        a_output = output_activation(z_output, self.output_dim)

        return a_output

    def backward_pass(self, input_data: np.ndarray, output_gradient: np.ndarray) -> tuple[
        list[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
        a_hidden = [stable_sigmoid(layer.z) for layer in self.hidden_layers]
        a_hidden.insert(0, input_data)

        # Output layer gradients
        dA_prev, dWeights_output, dBias_output = self.output_layer.backward_pass(a_hidden[-1], output_gradient)

        # Backprop through hidden layers
        gradients = []
        for i in reversed(range(len(self.hidden_layers))):
            # Calculate gradient of the activation
            sig_deriv = sigmoid_derivative(a_hidden[i + 1])

            # Apply chain rule (element-wise multiplication with sigmoid derivative)
            dA_prev *= sig_deriv

            dA_prev, dWeights, dBias = self.hidden_layers[i].backward_pass(a_hidden[i], dA_prev)
            gradients.append((dWeights, dBias))

        gradients.reverse()  # To match the order of layers
        return gradients, (dWeights_output, dBias_output)

    def parameter_update(self, gradients: List[Tuple[np.ndarray, np.ndarray]],
                         output_gradients: Tuple[np.ndarray, np.ndarray]) -> None:
        # Update hidden layer parameters
        for i, (dWeights, dBias) in enumerate(gradients):
            self.hidden_layers[i].parameter_update(dWeights, dBias)
        # Update output layer parameters
        self.output_layer.parameter_update(*output_gradients)


def output_activation(output, output_dim):
    if output_dim > 1:
        return softmax(output)
    else:
        return stable_sigmoid(output)


def stable_sigmoid(x):
    "Compute sigmoid function in a way that avoids overflow."
    z = np.exp(-np.abs(x))
    sigmoid = np.where(x >= 0, 1 / (1 + z), z / (1 + z))
    return sigmoid


def sigmoid_derivative(a):
    return a * (1 - a)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Avoid overflow
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
