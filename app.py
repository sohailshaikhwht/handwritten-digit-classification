import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Handwritten Digit Classifier",
    page_icon="🔢",
    layout="centered"
)


# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #e8f5e9;
    text-align: center;
    margin-top: 20px;
}

.digit {
    font-size: 70px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# CREATE CNN MODEL
# -----------------------------------
# -----------------------------------
# CREATE CNN MODEL
# -----------------------------------

# Clear previous Keras session
tf.keras.backend.clear_session()

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


# -----------------------------------
# LOAD MODEL WEIGHTS
# -----------------------------------

model.load_weights(
    "digit_model.weights.h5"
)


# -----------------------------------
# SESSION STATE
# -----------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------------
# TITLE
# -----------------------------------

st.markdown(
    '<div class="title">🔢 Handwritten Digit Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Draw a digit from 0 to 9 using your mouse</div>',
    unsafe_allow_html=True
)


# -----------------------------------
# DRAWING CANVAS
# -----------------------------------

st.subheader("✏️ Draw Your Digit")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    return_image_data=True,
    key="canvas"
)


# -----------------------------------
# BUTTONS
# -----------------------------------

col1, col2 = st.columns(2)

with col1:

    predict_button = st.button(
        "🔮 Predict",
        use_container_width=True
    )

with col2:

    clear_button = st.button(
        "🧹 Clear",
        use_container_width=True
    )


# -----------------------------------
# CLEAR BUTTON
# -----------------------------------

if clear_button:

    st.session_state.history = []

    st.rerun()


# -----------------------------------
# PREDICTION
# -----------------------------------

if predict_button:

    if canvas_result.image_data is not None:

        # Get image
        image = canvas_result.image_data

        # Convert to grayscale
        image = Image.fromarray(
            image.astype("uint8")
        ).convert("L")

        # Resize to MNIST size
        image = image.resize((28, 28))

        # Convert to NumPy
        image = np.array(image)

        # Normalize
        image = image.astype("float32") / 255.0

        # Reshape
        image = image.reshape(
            1, 28, 28, 1
        )

        # Predict
        prediction = model.predict(
            image,
            verbose=0
        )[0]

        # Get predicted digit
        predicted_digit = np.argmax(
            prediction
        )

        # Get confidence
        confidence = (
            np.max(prediction) * 100
        )

        # Save history
        st.session_state.history.append({
            "digit": int(predicted_digit),
            "confidence": float(confidence)
        })


        # -----------------------------------
        # RESULT
        # -----------------------------------

        st.markdown(
            f"""
            <div class="result-box">

            <div>Predicted Digit</div>

            <div class="digit">
            {predicted_digit}
            </div>

            <div>
            Confidence: <b>{confidence:.2f}%</b>
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # -----------------------------------
        # ALL DIGIT CONFIDENCES
        # -----------------------------------

        st.subheader("📊 Prediction Confidence")

        for digit in range(10):

            probability = prediction[digit] * 100

            st.write(
                f"Digit {digit}: {probability:.2f}%"
            )

            st.progress(
                float(prediction[digit])
            )


    else:

        st.warning(
            "⚠️ Please draw a digit first."
        )


# -----------------------------------
# PREDICTION HISTORY
# -----------------------------------

st.subheader("🕘 Prediction History")

if len(st.session_state.history) == 0:

    st.info(
        "No predictions yet. Draw a digit and click Predict."
    )

else:

    for i, item in enumerate(
        reversed(st.session_state.history),
        start=1
    ):

        st.write(
            f"**{i}. Digit:** {item['digit']} "
            f"— **Confidence:** "
            f"{item['confidence']:.2f}%"
        )


# ==================================================
# MODEL PERFORMANCE
# ==================================================

st.markdown("---")

st.header("📈 Model Performance")

st.write(
    "The CNN model was evaluated using the MNIST test dataset."
)


# -----------------------------------
# LOAD TEST DATA
# -----------------------------------

@st.cache_data
def load_test_data():

    (_, _), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    x_test = (
        x_test.astype("float32") / 255.0
    )

    x_test = x_test[..., None]

    return x_test, y_test


x_test, y_test = load_test_data()


# -----------------------------------
# MODEL PREDICTIONS
# -----------------------------------

@st.cache_data
def calculate_predictions(_model, x_test):

    predictions = _model.predict(
        x_test,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    return predicted_labels


predicted_labels = calculate_predictions(
    model,
    x_test
)


# -----------------------------------
# CALCULATE ACCURACY
# -----------------------------------

accuracy = np.mean(
    predicted_labels == y_test
)

accuracy_percentage = accuracy * 100


# -----------------------------------
# DISPLAY ACCURACY
# -----------------------------------

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Test Accuracy",
        f"{accuracy_percentage:.2f}%"
    )

with col2:

    st.metric(
        "Test Images",
        "10,000"
    )


# -----------------------------------
# CONFUSION MATRIX
# -----------------------------------

st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    predicted_labels
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=np.arange(10)
)

disp.plot(
    ax=ax
)

ax.set_title(
    "MNIST Handwritten Digit Confusion Matrix"
)

ax.set_xlabel(
    "Predicted Digit"
)

ax.set_ylabel(
    "Actual Digit"
)

st.pyplot(fig)


# -----------------------------------
# EXPLANATION
# -----------------------------------

st.info(
    """
    **How to read the confusion matrix:**

    • Rows represent the actual digits.

    • Columns represent the predicted digits.

    • Values on the diagonal represent correct predictions.

    • Values outside the diagonal represent incorrect predictions.

    • Higher diagonal values indicate better model performance.
    """
)


# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")

st.caption(
    "Built using Python • TensorFlow • CNN • MNIST • Streamlit"
)