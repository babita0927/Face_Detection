import streamlit as st
import cv2
import datetime as dt
from time import sleep

st.header("Face Detection Using OpenCV")

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect(gray, frame):
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]  # crop face rectngle
        roi_color = frame[y:y + h, x:x + w]

    return frame


def run_face_detection():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        st.write('Unable to load camera.')
        return

    st.write("Let's gooo...")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.write("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = detect(gray, frame)
        cv2.imshow('Video', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if st.button("Can I detect your face ?"):
    run_face_detection()
