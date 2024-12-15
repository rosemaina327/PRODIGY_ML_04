import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
from tensorflow.keras.models import load_model


# Load the trained model
model = load_model("hand_gesture_model.h5")

# Gesture labels (update these based on your training labels)
gesture_labels = ["01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb", 
                  "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"]

print("Model loaded and ready for predictions.")

# Function to preprocess static images
def preprocess_image(img_path):
    """Preprocess an image for prediction."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot load image at {img_path}")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img_resized = cv2.resize(img_gray, (64, 64))  # Resize to match model input
    img_normalized = img_resized / 255.0  # Normalize pixel values
    img_expanded = np.expand_dims(img_normalized, axis=(0, -1))  # Add batch and channel dimensions
    return img_expanded

# Function to predict gesture for a static image
def predict_image(img_path):
    """Predict the gesture for a single image."""
    try:
        processed_img = preprocess_image(img_path)
        predictions = model.predict(processed_img)  # Model prediction
        predicted_class = np.argmax(predictions)  # Get class with highest probability
        predicted_label = gesture_labels[predicted_class]
        print(f"Prediction: {predicted_label}")
    except Exception as e:
        print(f"Error: {e}")

# Function to preprocess webcam frames
def preprocess_frame(frame):
    """Preprocess a video frame for prediction."""
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img_resized = cv2.resize(img_gray, (64, 64))  # Resize to match model input
    img_normalized = img_resized / 255.0  # Normalize pixel values
    img_expanded = np.expand_dims(img_normalized, axis=(0, -1))  # Add batch and channel dimensions
    return img_expanded

# Function for real-time webcam gesture recognition
def run_webcam():
    """Run real-time hand gesture recognition using webcam."""
    cap = cv2.VideoCapture(0)  # 0 means default webcam
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Failed to grab frame.")
            break

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Predict the gesture (suppress TensorFlow's verbose output)
        predictions = model.predict(processed_frame, verbose=0)
        predicted_class = np.argmax(predictions)
        predicted_label = gesture_labels[predicted_class]

        # Display the prediction on the video feed
        cv2.putText(frame, f"Prediction: {predicted_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Hand Gesture Recognition", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
1

# Main function to select mode: static image or webcam
def main():
    print("Choose a mode:")
    print("1. Predict a static image")
    print("2. Run real-time webcam recognition")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        img_path = input("Enter the path to the image: ").strip()
        predict_image(img_path)
    elif choice == "2":
        run_webcam()
    else:
        print("Invalid choice.Please enter 1 or 2.")

print(f"Debug: Your input was '{choice}'")

if __name__ == "__main__":
    main()
