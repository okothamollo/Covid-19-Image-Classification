import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.preprocessing import LabelBinarizer # Assuming enc is defined and fit
import cv2

# Set page title and layout
st.set_page_config(page_title="COVID-19 Image Classifier", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('my_model.keras')
    return model

model = load_model()

# Initialize LabelBinarizer (assuming the same classes and order as training)
# In a real deployment, you would save and load the LabelBinarizer as well, or define the classes directly.
enc_classes = ['Covid', 'Normal', 'Viral Pneumonia'] # Make sure these match your training labels
enc = LabelBinarizer()
enc.fit(enc_classes)

# Streamlit App Title
st.title("COVID-19 Chest X-ray Classification")
st.write("Upload a chest X-ray image to classify it as COVID-19, Normal, or Viral Pneumonia.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_bytes = uploaded_file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64)) # Resize to target size (64x64 as used in training)
    img = img.astype('float32') / 255.0 # Normalize pixel values
    img = np.expand_dims(img, axis=0) # Add batch dimension

    # Make prediction
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = enc.inverse_transform(predictions)[0] # Use inverse_transform to get the label string

    st.success(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {predictions[0][predicted_class_index]*100:.2f}%")

