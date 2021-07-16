import cv2 as cv
import imutils

cam = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier("face.xml")
eye_cascade = cv.CascadeClassifier("eye.xml")

while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
                                          flags=cv.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    eyes = eye_cascade.detectMultiScale(gray, 1.03, minNeighbors=10)
    for (x, y, w, h) in eyes:
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('img', frame)

    if cv.waitKey(1) == 27:
        break

cam.release()
cv.destroyAllWindows()
