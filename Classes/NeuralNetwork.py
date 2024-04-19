from typing import Tuple, List
import numpy as np
from Classes.LinearLayer import LinearLayer


class NeuralNetwork:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, hidden_layers: int, learning_rate: float = 0.1):
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
            current_input = stable_sigmoid(z)  # Activation function
        z_output = self.output_layer.forward_pass(current_input)
        a_output = stable_sigmoid(z_output)
        return a_output

    def backward_pass(self, input_data: np.ndarray, output_gradient: np.ndarray) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
        a_hidden = [stable_sigmoid(layer.z) for layer in self.hidden_layers]
        a_hidden.insert(0, input_data)

        # Output layer gradients
        dA_prev, dWeights_output, dBias_output = self.output_layer.backward_pass(a_hidden[-1], output_gradient)

        # Backprop through hidden layers
        gradients = []
        for i in reversed(range(len(self.hidden_layers))):
            dA_prev, dWeights, dBias = self.hidden_layers[i].backward_pass(a_hidden[i], dA_prev)
            gradients.append((dWeights, dBias))

        gradients.reverse()  # To match the order of layers
        return gradients, (dWeights_output, dBias_output)

    def parameter_update(self, gradients: List[Tuple[np.ndarray, np.ndarray]], output_gradients: Tuple[np.ndarray, np.ndarray]) -> None:
        # Update hidden layer parameters
        for i, (dWeights, dBias) in enumerate(gradients):
            self.hidden_layers[i].parameter_update(dWeights, dBias)
        # Update output layer parameters
        self.output_layer.parameter_update(*output_gradients)


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
