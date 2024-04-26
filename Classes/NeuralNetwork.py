from typing import Tuple, List
import numpy as np
from Classes.LinearLayer import LinearLayer


class NeuralNetwork:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, hidden_layers: int,
                 learning_rate: float = 0.1):
        """
        Initializes a new neural network with specified dimensions and learning rate.

        Parameters:
        - input_dim: The size of the input data.
        - hidden_dim: The size of each hidden layer.
        - output_dim: The size of the output layer.
        - hidden_layers: The number of hidden layers in the network.
        - learning_rate: The step size used for updating the weights during training.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Initialize hidden layers: First hidden layer gets input_dim, subsequent get hidden_dim
        self.hidden_layers = [LinearLayer(hidden_dim, hidden_dim, learning_rate) for _ in range(hidden_layers)]
        self.hidden_layers[0] = LinearLayer(input_dim, hidden_dim, learning_rate)

        # Initialize the output layer
        self.output_layer = LinearLayer(hidden_dim, output_dim, learning_rate)

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        current_input = input_data
        # Process each layer's output through a stable sigmoid function
        for layer in self.hidden_layers:
            z = layer.forward_pass(current_input)
            current_input = stable_sigmoid(z)

        # Get the final layer's output and apply the appropriate output activation function
        z_output = self.output_layer.forward_pass(current_input)
        a_output = output_activation(z_output, self.output_dim)

        return a_output

    def backward_pass(self, input_data: np.ndarray, output_gradient: np.ndarray):
        a_hidden = [stable_sigmoid(layer.z) for layer in self.hidden_layers]
        a_hidden.insert(0, input_data)  # Include input data as the first activation for gradient computation

        # Calculate gradients for the output layer
        dA_prev, dWeights_output, dBias_output = self.output_layer.backward_pass(a_hidden[-1], output_gradient)

        gradients = []
        # Backpropagate through hidden layers
        for i in reversed(range(len(self.hidden_layers))):
            sig_deriv = sigmoid_derivative(a_hidden[i + 1])  # Derivative of sigmoid activation
            dA_prev *= sig_deriv  # Chain rule application

            dA_prev, dWeights, dBias = self.hidden_layers[i].backward_pass(a_hidden[i], dA_prev)
            gradients.append((dWeights, dBias))

        gradients.reverse()  # Match the layer order for updates
        return gradients, (dWeights_output, dBias_output)

    def parameter_update(self, gradients: list[tuple[np.ndarray, np.ndarray]],
                         output_gradients: tuple[np.ndarray, np.ndarray]) -> None:
        # Update parameters for each hidden layer
        for i, (dWeights, dBias) in enumerate(gradients):
            self.hidden_layers[i].parameter_update(dWeights, dBias)

        # Update parameters for the output layer
        self.output_layer.parameter_update(*output_gradients)


def output_activation(output, output_dim):
    if output_dim > 1:
        return softmax(output)
    else:
        return stable_sigmoid(output)


def stable_sigmoid(x):
    z = np.exp(-np.abs(x))
    sigmoid = np.where(x >= 0, 1 / (1 + z), z / (1 + z))
    return sigmoid


def sigmoid_derivative(a):
    return a * (1 - a)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Avoid overflow
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
