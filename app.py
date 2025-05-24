import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import joblib

# Load models
def load_models():
    # Load CNN
    with open("models/cnn_model.json", "r") as json_file:
        cnn_model = model_from_json(json_file.read())
    cnn_model.load_weights("models/cnn_weights.h5")
    
    # Create feature extractor
    feature_extractor = Model(
        inputs=cnn_model.layers[0].input,
        outputs=cnn_model.get_layer('feature_layer').output
    )
    
    # Load SVM
    svm = joblib.load("models/svm_model.pkl")
    
    return feature_extractor, svm

# Emotion prediction function
def predict_emotion(image):
    feature_extractor, svm = load_models()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    predictions = []
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))  # Shape: (1, 48, 48, 1)
        
        # Extract features and predict
        features = feature_extractor.predict(roi)
        pred = svm.predict(features)[0]
        predictions.append((x, y, w, h, pred))
    
    return predictions

# Streamlit UI
st.title("Real-Time Emotion Detection 🎭")
st.write("Upload an image with faces to detect emotions!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Get predictions
    predictions = predict_emotion(image)
    
    # Draw results
    emotion_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']
    for (x, y, w, h, pred) in predictions:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, emotion_labels[pred], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display
    st.image(image, channels="BGR", use_column_width=True)
