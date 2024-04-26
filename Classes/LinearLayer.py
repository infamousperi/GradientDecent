import numpy as np


class LinearLayer:
    def __init__(self, input_size: int, output_size: int, learning_rate: float = 0.1) -> None:
        self.z = None  # Placeholder for storing the linear transformation output (before activation).
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Xavier/Glorot initialization helps in keeping the signal in a reasonable range of values through many layers.
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / (input_size + output_size))
        self.bias = np.zeros((output_size,))

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        # Compute the linear part of the layer
        output_data = np.dot(input_data, self.weights.T) + self.bias
        self.z = output_data  # Store the linear transformation result used in backpropagation

        return output_data

    def backward_pass(self, input_data: np.ndarray, output_gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Gradient of the loss with respect to the inputs
        dA = np.dot(output_gradient, self.weights)
        # Gradient of the loss with respect to the weights
        weight_gradient = np.dot(output_gradient.T, input_data)
        # Gradient of the loss with respect to the biases, summing over the batch dimension
        bias_gradient = np.sum(output_gradient, axis=0)

        return dA, weight_gradient, bias_gradient

    def parameter_update(self, weight_gradient: np.ndarray, bias_gradient: np.ndarray) -> None:
        # Update weights and biases using gradient descent
        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient
