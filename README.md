# Cataract Detection using CNN

This project implements a deep learning-based approach to detect cataracts from eye images using Convolutional Neural Networks (CNNs). It is designed and run in a Kaggle environment using TensorFlow and Keras.

## 🧠 Objective

To develop an automated system that can classify eye images as having cataracts or not, enabling quicker diagnosis and screening using image processing and deep learning.

## 📁 Dataset Structure

## 📁 Dataset Structure

The dataset is organized as follows:

    /processed_images/
    ├── train/
    │   ├── cataract/
    │   └── normal/
    └── test/
        ├── cataract/
        └── normal/

Each image is labeled based on the folder name. The dataset is loaded from the Kaggle input directory and visualized using random samples.

## 🧰 Technologies & Libraries Used

- Python 3
- TensorFlow / Keras
- Matplotlib
- PIL
- NumPy
- Pandas
- ImageDataGenerator (for real-time data preprocessing)

## 🛠️ Model Architecture

The model is a simple CNN built with:

- Convolutional layers (`Conv2D`)
- Pooling layers (`MaxPooling2D`)
- Dense layers
- Dropout layers (to reduce overfitting)
- Optimizer: `Adam`
- Loss function: `categorical_crossentropy`

## 📈 Training Process

- Images are rescaled to normalize pixel values.
- The dataset is fed into a CNN using `ImageDataGenerator`.
- The model is trained on the training dataset and validated for accuracy and loss.

## 📊 Evaluation

The training metrics include:

- Accuracy over epochs
- Loss over epochs

These are visualized using `matplotlib`.

## 🖼️ Visualization

The notebook also displays a few randomly selected eye images along with their labels (cataract/normal) to give a sense of the dataset quality.

## 🚀 How to Run

To run the notebook:

1. Open the notebook in [Kaggle Notebooks](https://www.kaggle.com/code).
2. Make sure the dataset is uploaded and linked in the input directory.
3. Run all cells to preprocess data, train the model, and evaluate it.

## 📝 Future Improvements

- Use data augmentation to boost model robustness.
- Experiment with pre-trained models like VGG16, ResNet for transfer learning.
- Deploy the model in a web app using Flask or Streamlit.

## 📌 Author

**Krish Patel**  
For queries, reach out via email or GitHub.
