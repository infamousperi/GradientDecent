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

    # Now plot each group in a separate figure
    for lr, results in results_by_lr.items():
        plt.figure(figsize=(8, 6))
        for res in results:
            plt.plot(res['epochs'], res['train_loss'], marker='o', linestyle='-',
                     label=f"Hidden Size: {res['hidden_layer_size']}")
        plt.title(f'Training Loss Over Epochs\nLearning Rate: {lr}')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend(title="Hidden Layer Sizes")
        plt.grid(True)
        plt.show()