import numpy as np


class LinearLayer:
    def __init__(self, input_size: int, output_size: int, learning_rate: float = 0.1) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights with Xavier initialization for better convergence
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / (input_size + output_size))
        self.bias = np.zeros((output_size,))

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        output_data = np.dot(input_data, self.weights.T) + self.bias

        return output_data

    def backward_pass(self, input_data: np.ndarray, output_gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dA = np.matmul(output_gradient, self.weights)  # Vectorized matrix multiplication
        weight_gradient = np.matmul(output_gradient.T, input_data)  # Vectorized matrix multiplication
        bias_gradient = np.sum(output_gradient, axis=0)  # Efficient sum across batches

        return dA, weight_gradient, bias_gradient

    def parameter_update(self, weight_gradient: np.ndarray, bias_gradient: np.ndarray) -> None:
        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient
