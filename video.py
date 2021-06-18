import cv2 as cv
import imutils

cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()

    cv.imshow("video", frame)

    # convert the image to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # canny edge detection algorithm:
    #   noise reduction using Gaussian filtering
    #   finding edge gradient and direction for each pixel
    #   non-maximum suppression, removing unwanted "edges" by checking for local maximums
    #   hysteresis thresholding, takes in two threshold values, and only keeps edges with
    #       an intensity gradient that's in between
    edged = cv.Canny(gray, 30, 150)

    cv.imshow("canny", edged)

    # threshold image by setting all pixel values less than 225 to 255 and all pixel values >= 225 to 255
    thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)[1]
    cv.imshow("threshold", thresh)

    # contours (outlines)
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    output = frame.copy()

    # loop over the contours
    for c in contours:
        # draw each contour on the output image using a 3px thick red outline, then display the contours
        cv.drawContours(output, [c], -1, (0, 0, 255), 3)
        cv.imshow("contours", output)

    if cv.waitKey(1) == 27:
        break

cam.release()
cv.destroyAllWindows()
