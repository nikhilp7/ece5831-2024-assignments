import argparse
import matplotlib.pyplot as plt
import numpy as np
from mnist_data import MnistData

def main():
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(description="Display image from MNIST dataset")
    parser.add_argument('dataset_type', choices=['train', 'test'], help="Specify 'train' or 'test' dataset.")
    parser.add_argument('index', type=int, help="Image Index.")

    # Extract arguments provided through the command line
    args = parser.parse_args()  # Argparser used to handle input arguments
    dataset_type = args.dataset_type
    index = args.index

    # Load the MNIST dataset using MnistData class
    print("Loading MNIST dataset...")
    mnist = MnistData()
    (train_images, train_labels), (test_images, test_labels) = mnist.load()
    print("MNIST dataset loaded successfully.")

    # Choose the appropriate dataset based on the user's input
    print(f"Selecting the {dataset_type} dataset...")
    images = train_images if dataset_type == 'train' else test_images
    labels = train_labels if dataset_type == 'train' else test_labels

    # Ensure the provided index is valid within the dataset's bounds
    if index < 0 or index >= len(images):
        print(f"Error: Index {index} is out of range for the {dataset_type} dataset.")
        return
    print(f"Index {index} is valid for the {dataset_type} dataset.")

    # Reshape the selected image for visualization
    image = images[index].reshape(28, 28)
    label = np.argmax(labels[index])  # Assuming the labels are in one-hot encoded format

    # Display the image using matplotlib
    print(f"Displaying the {dataset_type} image at index {index} with label: {label}")
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()

    # Print the corresponding label for the displayed image
    print(f'The label for the {dataset_type} image at index {index} is: {label}')

# Script entry point
if __name__ == '__main__':
    main()
