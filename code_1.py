import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set dataset directory path
dataset_dir = 'D:/Vs Code tasks/ML/Hand Gesture/leapGestRecog'  # Update this path

# Load data
print("Loading Data...")
def load_data(dataset_dir, img_size=(64, 64)):
    X, y = [], []   
    for session_folder in os.listdir(dataset_dir):
        session_path = os.path.join(dataset_dir, session_folder)
        if not os.path.isdir(session_path):
            print(f"Skipping non-folder item: {session_folder}")
            continue
        for gesture_folder in os.listdir(session_path):
            gesture_path = os.path.join(session_path, gesture_folder)
            if not os.path.isdir(gesture_path):
                print(f"  Skipping non-folder item: {gesture_folder}")
                continue
            label = gesture_folder
            for img_file in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_file)
                # Skip non-image files
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"Skipping non-image file: {img_file}")
                    continue
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Skipping unreadable file: {img_path}")
                    continue
                # Preprocess image
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img_gray, img_size) / 255.0
                X.append(img_resized)
                y.append(label)
    return np.array(X), np.array(y)


print("Loading data...")
X, y = load_data(dataset_dir)
print(f"Loaded {len(X)} images with {len(set(y))} unique labels.")

# Encode labels and one-hot encode
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_one_hot = to_categorical(y_encoded)

# Reshape X for CNN input
X = X.reshape(-1, 64, 64, 1)

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Build CNN model
print("Building CNN Model")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Classification report and confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Save the model
model.save("hand_gesture_model.h5")
print("Model saved as hand_gesture_model.h5")
