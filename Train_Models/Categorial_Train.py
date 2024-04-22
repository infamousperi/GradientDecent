import numpy as np
from Classes.NeuralNetwork import NeuralNetwork


def softmax(model_outputs):
    exps = np.exp(model_outputs - np.max(model_outputs, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)


def categorical_cross_entropy(predictions, targets, epsilon=1e-10):
    # Clip the predictions to prevent log(0) error
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    # Calculate cross-entropy
    return -np.sum(targets * np.log(predictions)) / predictions.shape[0]


def categorical_compute_accuracy(predictions, labels):
    # Predictions are the output probabilities from softmax, labels are one-hot encoded.
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    accuracy = np.mean(pred_classes == true_classes)
    return accuracy


def categorical_train_model(model, train_images, train_labels, test_images, test_labels, epochs, patience, batch_size):
    best_test_loss = float('inf')
    wait = 0
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

    for epoch in range(epochs):
        # Shuffle the training data
        permutation = np.random.permutation(train_images.shape[0])
        train_images_shuffled = train_images[permutation]
        train_labels_shuffled = train_labels[permutation]

        # Mini-batch gradient descent
        for i in range(0, train_images.shape[0], batch_size):
            batch_train_images = train_images_shuffled[i:i + batch_size]
            batch_train_labels = train_labels_shuffled[i:i + batch_size]

            # Forward pass
            outputs = model.forward_pass(batch_train_images)
            predictions = softmax(outputs)
            train_loss = categorical_cross_entropy(predictions, batch_train_labels)
            error_propagation = predictions - batch_train_labels
            gradients = model.backward_pass(batch_train_images, error_propagation)

            # Update model parameters
            model.parameter_update(*gradients)

        # Evaluation on the entire training data
        train_outputs = model.forward_pass(train_images)
        train_predictions = softmax(train_outputs)
        train_loss = categorical_cross_entropy(train_predictions, train_labels)
        train_accuracy = categorical_compute_accuracy(train_predictions, train_labels)

        # Evaluation on the entire test data
        test_outputs = model.forward_pass(test_images)
        test_predictions = softmax(test_outputs)
        test_loss = categorical_cross_entropy(test_predictions, test_labels)
        test_accuracy = categorical_compute_accuracy(test_predictions, test_labels)

        # Store metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # Print progress
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Test Loss: {test_loss}, '
              f'Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}')

        # Early stopping based on test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            wait = 0  # Reset wait counter
        else:
            wait += 1
        if wait >= patience:
            print(f"Stopping early due to lack of improvement in test loss at epoch {epoch + 1}.")
            break

    return train_losses, test_losses, train_accuracies, test_accuracies


def categorical_evaluate_combinations(train_images, train_labels, test_images, test_labels, epochs, patience,
                                      batch_size, learning_rates, hidden_layer_sizes):
    results = []

    for lr in learning_rates:
        for hidden_size in hidden_layer_sizes:
            print(f"Evaluating model with learning rate {lr} and hidden layer size {hidden_size}")

            # Initialize model with current combination
            model = NeuralNetwork(input_dim=784, hidden_dim=hidden_size, output_dim=10, hidden_layers=3,
                                  learning_rate=lr)

            # Train model using the provided train_model function
            train_loss, test_loss, train_accuracy, test_accuracy = categorical_train_model(
                model, train_images, train_labels, test_images, test_labels, epochs, patience, batch_size
            )

            # Store the training process results for this combination
            results.append({
                "learning_rate": lr,
                "hidden_layer_size": hidden_size,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "epochs": list(range(1, len(test_loss) + 1))  # Generating a list of epoch numbers
            })

            # Optionally print out results for this combination
            print(f"Completed training for learning rate {lr} and hidden layer size {hidden_size}\n")

    return results