import matplotlib.pyplot as plt
import numpy as np


def show_hist(data, title):
    # Define the number of bins to match the number of digit categories, assuming digits 0-9
    bins = range(int(min(data)), int(max(data)) + 2)  # +2 to include the last edge

    # Plot histogram with specified bins
    plt.hist(data, bins=bins, edgecolor='black', align='left')

    # Set x-axis and y-axis labels
    plt.xlabel('Digit')
    plt.ylabel('Number of Images')

    # Set x-ticks to be centered on each bin for digits
    plt.xticks(range(int(min(data)), int(max(data)) + 1))  # Ensure all digit labels are shown

    # Set the title of the histogram
    plt.title(title)

    # Display the plot
    plt.show()


def show_image_matrix(images, labels, title):
    # Create a 10x10 grid of subplots with a fixed size
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(8, 8))

    # Set the title of the figure to describe the content, customizable by the user
    fig.suptitle('Visual Grid of Image Labels ' + title)

    # Iterate over each subplot (10 rows and 10 columns)
    for i in range(10):
        for j in range(10):
            # Find indices of images with label corresponding to the current row number
            matching_indices = np.where(labels == i)[0]

            # Check if there are any images with the current label
            if len(matching_indices) > 0:
                # Randomly pick an index and use that image
                idx = np.random.choice(matching_indices)
                image = images[idx]
            else:
                # Use a blank image if no images are found for this label
                image = np.zeros_like(images[0])

            # Display the image on the subplot, using grayscale color map
            axes[i, j].imshow(image, cmap='gray_r')
            axes[i, j].set_xticks([])  # Remove x-axis tick marks
            axes[i, j].set_yticks([])  # Remove y-axis tick marks

            # Add an axis label on the left side of the first column
            if j == 0:
                axes[i, j].set_ylabel(f'Label {i}', rotation=0, size='large', labelpad=30)

    # Adjust layout to make space for subplot titles and main title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
        # Create a figure with four subplots (2 rows, 2 columns)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))  # Adjusting figsize for four plots

        # Colors for consistency across plots
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

        # Plot training loss on the first subplot
        for res, color in zip(results, colors):
            ax1.plot(res['epochs'], res['train_loss'], marker='o', linestyle='-', color=color,
                     label=f"Hidden Size: {res['hidden_layer_size']}")
        ax1.set_title(f'Training Loss Over Epochs\nLearning Rate: {lr}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.legend(title="Hidden Layer Sizes")
        ax1.grid(True)

        # Plot training accuracy on the second subplot
        for res, color in zip(results, colors):
            ax2.plot(res['epochs'], res['train_accuracy'], marker='o', linestyle='-', color=color,
                     label=f"Hidden Size: {res['hidden_layer_size']}")
        ax2.set_title(f'Training Accuracy Over Epochs\nLearning Rate: {lr}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Training Accuracy')
        ax2.legend(title="Hidden Layer Sizes")
        ax2.grid(True)

        # Plot test loss on the third subplot
        for res, color in zip(results, colors):
            ax3.plot(res['epochs'], res['test_loss'], marker='o', linestyle='-', color=color,
                     label=f"Hidden Size: {res['hidden_layer_size']}")
        ax3.set_title(f'Test Loss Over Epochs\nLearning Rate: {lr}')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Test Loss')
        ax3.legend(title="Hidden Layer Sizes")
        ax3.grid(True)

        # Plot test accuracy on the fourth subplot
        for res, color in zip(results, colors):
            ax4.plot(res['epochs'], res['test_accuracy'], marker='o', linestyle='-', color=color,
                     label=f"Hidden Size: {res['hidden_layer_size']}")
        ax4.set_title(f'Test Accuracy Over Epochs\nLearning Rate: {lr}')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Test Accuracy')
        ax4.legend(title="Hidden Layer Sizes")
        ax4.grid(True)

        # Adding an overall title to the figure
        fig.suptitle(f'Hyperparameter-Evaluation with Learingrate {lr} and Hidden Size(4, 8, 16)', fontsize=16, fontweight='bold')

        # Show the complete figure with all four subplots
        plt.show()