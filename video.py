import cv2 as cv

cam = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cam.read()

    cv.imshow("video", frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # canny edge detection
    edged = cv.Canny(gray, 30, 150)
    cv.imshow("canny", edged)
    if cv.waitKey(1) == 27:
        break

cam.release()
cv.destroyAllWindows()
