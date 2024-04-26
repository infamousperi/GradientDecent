import numpy as np
from Classes.NeuralNetwork import NeuralNetwork


def binary_compute_accuracy(predictions, labels):
    predictions = predictions > 0.5  # Convert probabilities to binary predictions
    return np.mean(predictions == labels)  # Compute mean accuracy


def binary_cross_entropy(predictions, targets, epsilon=1e-10):
    # Clip predictions to avoid log(0) error. Clipping input to the log function to be between epsilon and 1-epsilon.
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    # Calculate the binary cross-entropy loss
    loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

    return loss


def binary_train_model(model, train_images, train_labels, test_images, test_labels, epochs, batch_size):
    # Lists to store the loss and accuracy metrics for both training and testing datasets
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

    # Loop over each epoch. An epoch is one full pass through the entire training dataset.
    for epoch in range(epochs):
        # Shuffle the training data to prevent the model from learning the order of the training data
        permutation = np.random.permutation(train_images.shape[0])
        train_images_shuffled = train_images[permutation]
        train_labels_shuffled = train_labels[permutation]

        # Mini-batch gradient descent
        for i in range(0, train_images.shape[0], batch_size):
            end = i + batch_size
            # Ensure the batch does not exceed the bounds of the array
            if end > train_images.shape[0]:
                end = train_images.shape[0]
            batch_train_images = train_images_shuffled[i:end]
            batch_train_labels = train_labels_shuffled[i:end]

            # Forward pass: compute predictions from the input data
            predictions = model.forward_pass(batch_train_images)
            # Compute the error as the difference between predictions and true labels
            error_propagation = predictions - batch_train_labels
            # Backward pass: compute gradients of the loss with respect to model parameters
            gradients = model.backward_pass(batch_train_images, error_propagation)

            # Update model parameters using the gradients computed
            hidden_gradients, output_gradients = gradients
            model.parameter_update(hidden_gradients, output_gradients)

        # Evaluate the model on the entire training dataset
        train_predictions = model.forward_pass(train_images)
        train_loss = binary_cross_entropy(train_predictions, train_labels)
        train_accuracy = binary_compute_accuracy(train_predictions, train_labels)

        # Evaluate the model on the entire testing dataset
        test_predictions = model.forward_pass(test_images)
        test_loss = binary_cross_entropy(test_predictions, test_labels)
        test_accuracy = binary_compute_accuracy(test_predictions, test_labels)

        # Append current epoch's metrics to lists
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # Print progress to keep track of training process
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Test Loss: {test_loss}, '
              f'Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}')

    return train_losses, test_losses, train_accuracies, test_accuracies


def binary_evaluate_combinations(train_images, train_labels, test_images, test_labels, epochs, batch_size,
                                 learning_rates, hidden_layer_sizes):
    results = []

    for lr in learning_rates:
        for hidden_size in hidden_layer_sizes:
            print(f"Evaluating model with learning rate {lr} and hidden layer size {hidden_size}")

            # Initialize model with current combination
            model = NeuralNetwork(input_dim=784, hidden_dim=hidden_size, output_dim=1, hidden_layers=1,
                                  learning_rate=lr)

            # Train model using the provided train_model function
            train_loss, test_loss, train_accuracy, test_accuracy = binary_train_model(
                model, train_images, train_labels, test_images, test_labels, epochs, batch_size
            )

            # Store the training process results for this combination
            results.append({
                "learning_rate": lr,
                "hidden_layer_size": hidden_size,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "epochs": list(range(1, len(test_loss) + 1))  # Generating a list of epoch numbers
            })

            # Optionally print out results for this combination
            print(f"Completed training for learning rate {lr} and hidden layer size {hidden_size}\n")

    return results
