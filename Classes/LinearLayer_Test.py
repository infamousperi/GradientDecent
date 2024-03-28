import unittest
import numpy as np
from Classes.LinearLayer import LinearLayer


class TestLinearLayer(unittest.TestCase):
    def test_forward_pass(self):
        # Simple input, weights, and bias
        # Fixed values for input, weights, and bias
        input_data = np.array([[1, 2], [3, 4]])
        weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        bias = np.array([[0.5], [1.0]])

        # Expected output
        A_expected = np.array([[1.0, 1.6], [2.1, 3.5]])

        # Create the linear layer
        layer = LinearLayer(2, 2)
        layer.weights = weights
        layer.bias = bias

        # Forward pass
        A = layer.forward_pass(input_data)

        # Compare with expected output
        np.testing.assert_allclose(A, A_expected)

    def test_backward_pass(self):
        # Input data from the forward pass
        input_data = np.array([[1, 2], [3, 4]])

        # Weights and bias (same as forward pass for consistency)
        weights = np.array([[0.5, 0.7], [0.3, 0.4]])
        bias = np.array([[0.1], [0.2]])

        # Expected output gradient (arbitrary for testing)
        output_gradient = np.array([[0.5, 0.6], [0.7, 0.8]])

        # Create the linear layer
        layer = LinearLayer(2, 2)
        layer.weights = weights
        layer.bias = bias

        # Forward pass (to get actual output)
        A = layer.forward_pass(input_data)

        # Backward pass
        dA, weight_gradient, bias_gradient = layer.backward_pass(A, output_gradient)

        # Expected input gradient (calculated using output_gradient and weights)
        expected_dA = np.array([[0.43, 0.59], [0.59, 0.81]])
        expected_weight_gradient = np.array([[4.15, 2.49], [4.8, 2.88]])
        expected_bias_gradient = np.array([1.2, 1.4])

        # Compare gradients
        np.testing.assert_allclose(dA, expected_dA)
        np.testing.assert_allclose(weight_gradient, expected_weight_gradient)
        np.testing.assert_allclose(bias_gradient, expected_bias_gradient)

    def test_parameter_update(self):
        # Learning rate
        learning_rate = 0.1

        # Sample weights and bias
        weights = np.array([[0.5, 0.7], [0.3, 0.4]])
        bias = np.array([[0.1], [0.2]])

        # Sample weight and bias gradients
        weight_gradient = np.array([[-0.2, 0.1], [0.3, -0.1]])
        bias_gradient = np.array([[-0.05], [0.03]])

        # Create the linear layer with initial parameters
        layer = LinearLayer(2, 2)
        layer.weights = weights.copy()  # Avoid modifying original array
        layer.bias = bias.copy()

        # Expected values
        expected_weights = weights - learning_rate * weight_gradient
        expected_bias = bias - learning_rate * bias_gradient

        # Perform parameter update
        layer.parameter_update(weight_gradient, bias_gradient)

        # Compare updated
        np.testing.assert_allclose(layer.weights, expected_weights)
        np.testing.assert_allclose(layer.bias, expected_bias)