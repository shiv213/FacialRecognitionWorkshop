import numpy as np
import cv2 as cv
import imutils
from imutils import face_utils
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[l_start:l_end]
        rightEye = shape[r_start:r_end]
        leftEyeHull = cv.convexHull(leftEye)
        rightEyeHull = cv.convexHull(rightEye)
        cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    cv.imshow("eyes", frame)
    key = cv.waitKey(1) & 0xFF

    if key == 27:
        break

cv.destroyAllWindows()

