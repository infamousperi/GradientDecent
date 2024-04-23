import matplotlib.pyplot as plt
import numpy as np


def show_hist(data):
    plt.hist(data, edgecolor='black')
    plt.xlabel('Digit')
    plt.ylabel('Number of images')
    plt.show()


def show_image_matrix(images, labels):
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(8, 8))

    # Iterate over subplots and display corresponding images from train_images
    for i in range(10):
        for j in range(10):
            matching_indices = (labels == i).nonzero()[0]
            if len(matching_indices) > 0:
                idx = matching_indices[np.random.randint(len(matching_indices))]
                image = images[idx]
            else:
                # If no images for this digit, use a blank image
                image = np.zeros_like(images[0])

            # Display the image on the subplot
            axes[i, j].imshow(image, cmap='gray')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


def show_evaluation_results(e_results):
    results_by_lr = {}
    for result in e_results:
        lr = result['learning_rate']
        if lr not in results_by_lr:
            results_by_lr[lr] = []
        results_by_lr[lr].append(result)

    # Now plot each group in a separate figure with subplots for loss and accuracy
    for lr, results in results_by_lr.items():
        # Create a figure with two subplots (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Adjusting figsize to accommodate both plots

        # Plot training loss on the first subplot
        for res in results:
            ax1.plot(res['epochs'], res['train_loss'], marker='o', linestyle='-',
                     label=f"Hidden Size: {res['hidden_layer_size']}")
        ax1.set_title(f'Training Loss Over Epochs\nLearning Rate: {lr}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.legend(title="Hidden Layer Sizes")
        ax1.grid(True)

        # Plot training accuracy on the second subplot
        for res in results:
            ax2.plot(res['epochs'], res['train_accuracy'], marker='o', linestyle='-',
                     label=f"Hidden Size: {res['hidden_layer_size']}")
        ax2.set_title(f'Training Accuracy Over Epochs\nLearning Rate: {lr}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Training Accuracy')
        ax2.legend(title="Hidden Layer Sizes")
        ax2.grid(True)

        # Show the complete figure with both subplots
        plt.show()