import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# Load MNIST dataset
# -----------------------------

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Test images:", x_test.shape)

# Normalize images
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_test = x_test[..., None]


# -----------------------------
# Create CNN model
# -----------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        64, activation="relu"
    ),

    tf.keras.layers.Dense(
        10, activation="softmax"
    )
])


# -----------------------------
# Load trained weights
# -----------------------------

model.load_weights(
    "digit_model.weights.h5"
)

print("Model weights loaded successfully!")

# -----------------------------
# Compile model
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Evaluate model
# -----------------------------

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print()
print("Test Accuracy:", accuracy)
print("Test Loss:", loss)


# -----------------------------
# Make predictions
# -----------------------------

predictions = model.predict(
    x_test,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)


# -----------------------------
# Create confusion matrix
# -----------------------------

cm = confusion_matrix(
    y_test,
    predicted_labels
)


# -----------------------------
# Display confusion matrix
# -----------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=np.arange(10)
)

disp.plot()

plt.title(
    "MNIST Handwritten Digit Confusion Matrix"
)

plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")

plt.show()