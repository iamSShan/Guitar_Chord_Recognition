import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
from keras.saving import save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


# Image dimensions and configuration
IMG_WIDTH, IMG_HEIGHT = 200, 200
BATCH_SIZE = 64
# Chords dataset
DATASET_PATH = "data_prep/chords_dataset"


def build_cnn_model(width, height, num_classes):
    """
    Builds and compiles a convolutional neural network (CNN) for image classification.

    Parameters:
    - width (int): Width of the input images.
    - height (int): Height of the input images.
    - num_classes (int): Number of output classes for classification.

    Returns:
    - cnn (keras.Sequential): The compiled CNN model.
    - [checkpoint] (list): A list containing a ModelCheckpoint callback to save the best model.
    """

    # Initialize a sequential model
    cnn = Sequential()

    # First convolutional layer with 32 filters, 5x5 kernel size, ReLU activation
    # Takes input images with 1 color channel (grayscale)
    cnn.add(
        Conv2D(
            32, kernel_size=(5, 5), activation="relu", input_shape=(width, height, 1)
        )
    )
    # First max pooling layer with 2x2 pool size, stride 2, padding same
    cnn.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"))

    # Second convolutional layer with 64 filters, 5x5 kernel size, ReLU activation
    cnn.add(Conv2D(64, kernel_size=(5, 5), activation="relu"))
    # Second max pooling layer with 5x5 pool size, stride 5, padding same
    cnn.add(MaxPooling2D(pool_size=(5, 5), strides=5, padding="same"))

    # Flatten the 2D feature maps to 1D feature vectors
    cnn.add(Flatten())
    # Fully connected (dense) layer with 1024 neurons and ReLU activation
    cnn.add(Dense(1024, activation="relu"))

    # Dropout layer to reduce overfitting, drops 60% of the inputs
    cnn.add(Dropout(0.6))

    # Output layer with `num_classes` neurons and softmax activation for multi-class classification
    cnn.add(Dense(num_classes, activation="softmax"))

    # Compile the model with Adam optimizer and categorical crossentropy loss
    cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Define a callback to save the best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        filepath="guitar_model.h5",  # File to save the model
        monitor="val_accuracy",  # Metric to monitor
        verbose=1,  # Print messages when saving
        save_best_only=True,  # Only save the best model
        mode="max",  # Save when val_accuracy is at its maximum
    )

    return cnn, [checkpoint]


def train_cnn_model():
    """
    Prepares data generators, trains the CNN, evaluates its performance,
    and visualizes the results using a classification report and confusion matrix
    """

    # Set up an image data generator with real-time data augmentation and validation split
    data_augmentation = ImageDataGenerator(
        rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
        rotation_range=15,  # Random rotation within 15 degrees
        width_shift_range=0.2,  # Horizontal shift up to 20% of image width
        height_shift_range=0.2,  # Vertical shift up to 20% of image height
        shear_range=0.2,  # Shear intensity
        zoom_range=0.2,  # Zoom in/out up to 20%
        fill_mode="nearest",  # Fill missing pixels after transformation
        validation_split=0.2,  # Reserve 20% of data for validation
    )

    # Load and augment training data from directory
    train_data = data_augmentation.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_WIDTH, IMG_HEIGHT),  # Resize images to target size
        batch_size=BATCH_SIZE,
        color_mode="grayscale",  # Use grayscale images
        class_mode="categorical",  # One-hot encoded labels
        subset="training",  # Use training split
        seed=42,  # Seed for reproducibility
    )

    # Load and augment validation data using the same generator and parameters
    val_data = data_augmentation.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        subset="validation",
        seed=42,
    )

    # Build the CNN model using the image dimensions and number of classes
    model, callbacks = build_cnn_model(
        IMG_WIDTH, IMG_HEIGHT, num_classes=train_data.num_classes
    )

    # Train the model for 10 epochs with validation data and callbacks
    model.fit(train_data, epochs=10, validation_data=val_data, callbacks=callbacks)

    # Get ground truth and predictions
    val_data.reset()  # <- important before predicting with generator
    y_true = val_data.classes
    # Predict class probabilities for validation data
    y_pred_probs = model.predict(val_data, verbose=1)

    # Convert probabilities to predicted class indices
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Labels for display
    labels = list(val_data.class_indices.keys())

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Final evaluation (redundant with earlier one but sometimes used after predictions)
    evaluation = model.evaluate(val_data)
    print(f"Final Accuracy: {evaluation[1] * 100:.2f}%")
    print(f"Final Loss: {evaluation[0]:.4f}")

    # Print dictionary mapping class names to label indices
    print(train_data.class_indices)

    # model.save("guitar_model.h5")
    # save_model(model, 'guitar_model.keras')


if __name__ == "__main__":
    train_cnn_model()
