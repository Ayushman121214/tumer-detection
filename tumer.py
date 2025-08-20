import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# Load the trained model
model = keras.models.load_model(r"C:\Users\95324\OneDrive\Desktop\models\best_brain_tumor_model.keras")

# Function to preprocess image
def preprocess_image(image):
    """
    Convert image to RGB (if grayscale), resize to 224x224, normalize, and add batch dimension.
    """
    image = image.convert("RGB")  # Ensure the image has 3 channels (for RGB model)
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values to range [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension for model
    return image

# Class labels
class_labels = ['Glioma Tumor', 'Meningioma Tumor', 'Nos Tumor', 'Pituitary Tumor']

# Streamlit App Design
st.title("Brain Tumor Detection System")
st.write("Upload an MRI scan to predict the type of tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Open image
    st.image(image, caption='Uploaded Image', use_column_width=True)  # Display uploaded image

    # Prediction button
    if st.button("Predict"):
        processed_image = preprocess_image(image)  # Preprocess image
        prediction = model.predict(processed_image)  # Model prediction
        predicted_class = class_labels[np.argmax(prediction)]  # Get class label
        confidence = np.max(prediction) * 100  # Get confidence score

        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")
