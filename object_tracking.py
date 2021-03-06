import cv2 as cv
import imutils

cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=600)

    # defining boundaries or range for color (in HSV color space)
    lower = (19, 29, 0)
    upper = (88, 172, 255)

    # blurring image
    blurred = cv.GaussianBlur(frame, (11, 11), 0)

    # converting image to the HSV color space
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

    # constructing a mask for our color range
    mask = cv.inRange(hsv, lower, upper)

    # perform a series of dilations and erosions to remove any noise
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    # contours (outlines)
    contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    output = frame.copy()
    center = None

    # loop over the contours
    for c in contours:
        # draw each contour on the output image using a 3px thick red outline, then display the contours
        cv.drawContours(output, [c], -1, (0, 0, 255), 3)
        cv.imshow("contours", output)

    # check if there's at least one contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv.contourArea)     # find largest contour
        ((x, y), radius) = cv.minEnclosingCircle(largest_contour)
        M = cv.moments(largest_contour)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / int(M["m00"])))

        cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv.circle(frame, center, 5, (0, 0, 255), -1)

    cv.imshow("video", frame)

    if cv.waitKey(1) == 27:
        break

cam.release()
cv.destroyAllWindows()
