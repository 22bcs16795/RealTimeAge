import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

st.set_page_config(page_title="Real-Time Age Detection", page_icon="🎭")
st.title("🎭 Real-Time Age Detection with DeepFace")

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# History to smooth age predictions
age_history = []

st.write("Use your webcam and see your predicted age in real-time!")

# Webcam input
camera_input = st.camera_input("Capture your face")

if camera_input is not None:
    # Convert uploaded image to OpenCV format
    img = np.array(Image.open(camera_input))
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    age = None
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        try:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (224, 224))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            result = DeepFace.analyze(face_img, actions=['age'], enforce_detection=False)
            age = int(result[0]['age'])
        except Exception as e:
            st.write("DeepFace error:", e)
            age = None

        # Smooth age over last 5 predictions
        if age is not None:
            age_history.append(age)
            if len(age_history) > 5:
                age_history.pop(0)
            age_avg = sum(age_history) // len(age_history)
            cv2.putText(frame, f"Age: {age_avg}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if len(faces) == 0:
        st.write("Face not detected")

    # Convert back to RGB for display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame, caption="Real-Time Age Detection", use_column_width=True)
