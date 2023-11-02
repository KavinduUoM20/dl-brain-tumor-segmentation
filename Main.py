import os
import numpy as np
import cv2
import streamlit as st    # Streamlit run class.py
from tensorflow.keras.models import load_model

# Load your fitted model
model_path = 'trained_brain_tumor_model.h5'  # Change this to your actual model path
model = load_model(model_path)

# Define class labels
class_labels = ['category1_tumor', 'category2_tumor', 'category3_tumor', 'no_tumor']

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize image pixels
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def main():
    st.title("Tumor Classifier App")
    st.write("Upload an image and the app will predict whether it contains a tumor or not.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        preprocessed_image = preprocess_image(image)
        predictions = model.predict(preprocessed_image)
        predicted_class = class_labels[np.argmax(predictions)]
        
        st.write(f"Predicted Class: {predicted_class}")
        st.write("Class Probabilities:")
        for i, label in enumerate(class_labels):
            st.write(f"{label}: {predictions[0][i]:.4f}")

if __name__ == '__main__':
    main()
