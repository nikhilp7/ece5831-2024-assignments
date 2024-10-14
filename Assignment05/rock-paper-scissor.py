import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image
import cv2
import os
import argparse

# Load the model and the labels
model_path = os.path.join(os.path.dirname(__file__), "keras_model.h5")
labels_path = os.path.join(os.path.dirname(__file__), "labels.txt")

# Load the labels
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the pre-trained Teachable Machine model
model = tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    """
    Preprocess the input image for the model.
    
    This function opens the image file, converts it to RGB, resizes it to 224x224 
    (which is the size expected by the model), converts it to a NumPy array, 
    adds a batch dimension, and normalizes the image pixel values to a [0,1] range.
    
    Args:
    - image_path (str): Path to the image file to be classified.
    
    Returns:
    - img_array (numpy.ndarray): The preprocessed image ready for model input.
    """
    img = Image.open(image_path).convert('RGB')  # Open the image and convert to RGB
    img = img.resize((224, 224))  # Resize the image to the size expected by the model
    img_array = np.array(img)  # Convert the image to a NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

def classify_image(image_path):
    """
    Classify the image using the pre-trained model.
    
    This function takes the path to an image, preprocesses it, and then uses the 
    model to predict its class. It returns the predicted class label and the 
    confidence score.
    
    Args:
    - image_path (str): Path to the image file to be classified.
    
    Returns:
    - class_name (str): The predicted class label ('rock', 'paper', or 'scissors').
    - confidence (float): The confidence score of the prediction.
    """
    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Make predictions using the model
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])  # Get the index of the highest prediction
    confidence = predictions[0][class_index]  # Get the confidence score for the prediction
    
    return labels[class_index], confidence

if __name__ == "__main__":
    """
    Main function to handle command-line arguments and execute the image classification.
    
    This script accepts an image path from the command line, classifies the image 
    as 'rock', 'paper', or 'scissors', and displays the image with the predicted 
    class and confidence score using matplotlib.
    """
    
    # Setup argument parser to accept an image path from the command line
    parser = argparse.ArgumentParser(description='Classify an image as rock, paper, or scissors.')
    parser.add_argument('--image-path', required=True, help='Path to the image to classify')
    
    # Parse the arguments from the command line
    args = parser.parse_args()

    # Get the image path from the parsed arguments
    image_path = args.image_path

    # Classify the image using the pre-trained model
    class_name, confidence_score = classify_image(image_path)

    # Display the image using matplotlib
    img = cv2.imread(image_path)  # Read the image using OpenCV
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image from BGR (OpenCV format) to RGB
    plt.imshow(img_rgb)  # Display the image using matplotlib
    plt.axis('off')  # Hide the axis for better visualization
    plt.title(f"Class: {class_name}, Confidence: {confidence_score:.4f}")  # Show class and confidence in the title
    plt.show()  # Show the image

    # Output the result to the command line
    print(f"Class: {class_name}")
    print(f"Confidence Score: {confidence_score:.4f}")
