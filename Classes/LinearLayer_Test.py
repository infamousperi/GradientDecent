import unittest
import numpy as np
from Classes.LinearLayer import LinearLayer


class TestLinearLayer(unittest.TestCase):
    def test_forward_backward_update(self):
        # Fixed values for input, weights, biases and learning rate
        input_data = np.array([[1, 2], [3, 4]])
        weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        bias = np.array([0.5, 0.5])
        learning_rate = 0.01

        # Define example output_gradient
        output_gradient = np.array([[0.1, 0.2], [0.3, 0.4]])

        # Implementation
        layer = LinearLayer(2, 2, learning_rate=learning_rate)
        layer.weights = weights.copy()
        layer.bias = bias.copy()

        # Forward pass
        output_data = layer.forward_pass(input_data)
        expected_output_data = 1.5

        # Backward pass
        expected_input_gradient = 0
        expected_weight_gradient = 0
        expected_bias_gradient = 0
        input_gradient, weight_gradient, bias_gradient = layer.backward_pass(input_data, output_gradient)

        # Parameter update
        expected_weights = 0
        expected_bias = 0
        layer.parameter_update(weight_gradient, bias_gradient)

        # Test forward pass
        self.assertEqual(output_data, expected_output_data)

        # Test backward pass
        self.assertEqual(input_gradient, expected_input_gradient)
        self.assertEqual(weight_gradient, expected_weight_gradient)
        self.assertEqual(bias_gradient, expected_bias_gradient)

        # Test update
        self.assertEqual(layer.weights, expected_weights)
        self.assertEqual(layer.bias, expected_bias)
