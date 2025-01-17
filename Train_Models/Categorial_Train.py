import numpy as np
from Classes.NeuralNetwork import NeuralNetwork


def categorical_compute_accuracy(predictions, labels):
    predictions = np.argmax(predictions, axis=1)  # Select the class with the highest probability
    labels = np.argmax(labels, axis=1)  # Assuming labels are one-hot encoded
    return np.mean(predictions == labels)  # Compute mean accuracy


def softmax_cross_entropy(predictions, targets, epsilon=1e-10):
    # Apply softmax to predictions
    exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    softmax_preds = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    # Clip predictions to avoid log(0) error
    softmax_preds = np.clip(softmax_preds, epsilon, 1 - epsilon)
    # Calculate the cross-entropy loss
    loss = -np.mean(np.sum(targets * np.log(softmax_preds), axis=1))
    return loss


def categorical_train_model(model, train_images, train_labels, test_images, test_labels, epochs, batch_size):
    # Initialize lists to store metrics for each epoch
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

    for epoch in range(epochs):
        # Randomly shuffle the training data at the beginning of each epoch to prevent bias
        permutation = np.random.permutation(train_images.shape[0])
        train_images_shuffled = train_images[permutation]
        train_labels_shuffled = train_labels[permutation]

        # Mini-batch gradient descent
        for i in range(0, train_images.shape[0], batch_size):
            end = i + batch_size
            # Ensure the batch does not exceed the number of training samples
            if end > train_images.shape[0]:
                end = train_images.shape[0]
            batch_train_images = train_images_shuffled[i:end]
            batch_train_labels = train_labels_shuffled[i:end]

            # Conduct forward and backward pass to update model parameters
            predictions = model.forward_pass(batch_train_images)
            error_propagation = predictions - batch_train_labels
            gradients = model.backward_pass(batch_train_images, error_propagation)

            # Update the model's weights and biases based on computed gradients
            hidden_gradients, output_gradients = gradients
            model.parameter_update(hidden_gradients, output_gradients)

        # Evaluation of the model on the entire training set
        train_predictions = model.forward_pass(train_images)
        train_loss = softmax_cross_entropy(train_predictions, train_labels)
        train_accuracy = categorical_compute_accuracy(train_predictions, train_labels)

        # Evaluation of the model on the entire test set
        test_predictions = model.forward_pass(test_images)
        test_loss = softmax_cross_entropy(test_predictions, test_labels)
        test_accuracy = categorical_compute_accuracy(test_predictions, test_labels)

        # Record performance metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # Output current epoch's performance metrics
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Test Loss: {test_loss}, '
              f'Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}')

    return train_losses, test_losses, train_accuracies, test_accuracies


def categorical_evaluate_combinations(train_images, train_labels, test_images, test_labels, epochs,
                                      batch_size,
                                      learning_rates, hidden_layer_sizes):
    results = []

    for lr in learning_rates:
        for hidden_size in hidden_layer_sizes:
            print(f"Evaluating model with learning rate {lr} and hidden layer size {hidden_size}")

            # Initialize model with current combination
            model = NeuralNetwork(input_dim=784, hidden_dim=hidden_size, output_dim=10, hidden_layers=3,
                                  learning_rate=lr)

            # Train model using the provided train_model function
            train_loss, test_loss, train_accuracy, test_accuracy = categorical_train_model(
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
