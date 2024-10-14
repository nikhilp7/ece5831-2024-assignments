import cv2
import numpy as np
import tensorflow as tf

def load_labels(label_file):
    """
    Load class names from the labels.txt file.

    This function reads a text file containing the class names (rock, paper, scissors),
    one per line, and returns them as a list.

    Args:
    - label_file (str): Path to the labels.txt file.

    Returns:
    - class_names (list of str): A list of class names (rock, paper, scissors).
    """
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

# Load the trained model from the Teachable Machine
model = tf.keras.models.load_model('keras_model.h5')

# Load the class names from the labels.txt file
class_names = load_labels('labels.txt')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    """
    Main loop for capturing video frames and making predictions in real-time.

    The loop continuously captures frames from the webcam, preprocesses them to match
    the model input format, and then uses the pre-trained model to predict whether the
    hand gesture is 'rock', 'paper', or 'scissors'. The predicted label is displayed on
    the frame, and the video is shown in a window.

    The loop runs until the 'q' key is pressed.
    """
    
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the image for model prediction
    # Resize the frame to 224x224 (the input size of the model)
    img = cv2.resize(frame, (224, 224))
    # Convert the image to a numpy array and normalize pixel values to the range [0, 1]
    img = np.array(img, dtype=np.float32) / 255.0
    # Add an extra dimension to the array to represent the batch size (required by the model)
    img = np.expand_dims(img, axis=0)

    # Predict the class of the image using the model
    predictions = model.predict(img)
    # Find the index of the highest prediction score (the most likely class)
    class_idx = np.argmax(predictions)
    # Get the class label corresponding to the index
    prediction_label = class_names[class_idx]
    
    # Display the resulting frame with the predicted class label
    # The predicted label is added as text at the top-left corner of the frame
    cv2.putText(frame, f'Prediction: {prediction_label}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Show the video feed with the prediction in a window named 'Rock Paper Scissors'
    cv2.imshow('Rock Paper Scissors', frame)

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows when the loop is finished
cap.release()
cv2.destroyAllWindows()
