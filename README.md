# ✍️ Handwritten Digit Classification

A Deep Learning project that recognizes handwritten digits (0–9) using a **Convolutional Neural Network (CNN)** trained on the MNIST dataset. The project also includes an interactive web interface built with Streamlit, allowing users to experience handwritten digit prediction.

## 🚀 Project Overview

Handwritten Digit Classification is an image recognition application that uses deep learning to identify handwritten numerical digits.

The CNN model learns patterns and features from handwritten digit images and predicts which digit (0–9) is represented.

## ✨ Features

* **Digit Recognition:** Predicts handwritten digits from 0 to 9.
* **CNN Model:** Uses a Convolutional Neural Network for image classification.
* **MNIST Dataset:** Trained using the popular MNIST handwritten digit dataset.
* **Streamlit Interface:** Provides an interactive web application.
* **Model Evaluation:** Includes a separate script to evaluate model performance.
* **Pre-trained Weights:** Includes saved model weights for prediction.

## 🛠️ Technologies Used

| Technology         | Purpose                   |
| ------------------ | ------------------------- |
| Python             | Programming language      |
| TensorFlow / Keras | Deep learning framework   |
| CNN                | Image classification      |
| MNIST              | Handwritten digit dataset |
| NumPy              | Numerical operations      |
| Streamlit          | Web application interface |

## 📂 Project Structure

```text
handwritten-digit-classification/
│
├── app.py
├── train_model.py
├── evaluate_model.py
├── digit_model.weights.h5
├── requirements.txt
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sohailshaikhwht/handwritten-digit-classification.git
```

### 2. Navigate to the Project Folder

```bash
cd handwritten-digit-classification
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 🧠 Model Training

To train the CNN model using the MNIST dataset, run:

```bash
python train_model.py
```

## 📊 Model Evaluation

To evaluate the trained model, execute:

```bash
python evaluate_model.py
```

## 🔍 How It Works

1. The MNIST dataset provides handwritten digit images.
2. Images are preprocessed before training.
3. A CNN model learns important image features.
4. The trained model identifies patterns in handwritten digits.
5. The application uses the trained weights to generate predictions.
6. The predicted digit is displayed through the Streamlit interface.

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

* Deep Learning and Neural Networks.
* Convolutional Neural Networks (CNN).
* Image preprocessing and classification.
* Working with the MNIST dataset.
* TensorFlow and Keras.
* Building interactive applications using Streamlit.
* Model training and evaluation.

## 🔮 Future Improvements

* Add a drawing canvas for handwritten digit input.
* Display prediction confidence scores.
* Improve model accuracy through hyperparameter tuning.
* Support image uploads for digit recognition.
* Deploy the application online for public access.

## 👨‍💻 Author

**Sohail Shaikh**

GitHub: [sohailshaikhwht](https://github.com/sohailshaikhwht)

Project Repository: [Handwritten Digit Classification](https://github.com/sohailshaikhwht/handwritten-digit-classification)

---

⭐ If you find this project interesting, consider giving the repository a star!
