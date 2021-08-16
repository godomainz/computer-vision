import cv2

# Loading the Cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Defining a function that will do the directions
def detect(grey, frame):
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)