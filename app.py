import streamlit as st
import cv2
import numpy as np
from your_model_file import load_model, predict_emotion

st.title("Real-Time Emotion Detection")
model = load_model()  # Your model loading function

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
if uploaded_file:
    image = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    emotion = predict_emotion(model, image)
    st.image(image, caption=f'Predicted Emotion: {emotion}')