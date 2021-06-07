import numpy as np
import matplotlib
import cv2 as cv

# load an image
img = cv.imread("red.png")

# images are represented as a multi-dimensional Numpy array
# shape: height=rows, width=columns, depth=number of channels
(h, w, d) = img.shape
# print out dimensions of image
print("width={}, height={}, depth={}".format(w, h, d))

# get rgb value of pixel at x=100, y=50 and print it out
(B, G, R) = img[50, 100]
print("R={}, G={}, B={}".format(R, G, B))

while True:
    # extract a 50 by 50 pixel square ROI (region of interest) from the input image starting at x=40, y=60
    roi = img[60:110, 40:90]
    cv.imshow("ROI", roi)

    # resize image
    resized = cv.resize(img, (500, 500))
    cv.imshow("resized", resized)

    # rotate image 45 deg clockwise
    # compute the center of the image, construct rotation matrix, then apply affine warp
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, -45, 1.0)
    rotated = cv.warpAffine(img, M, (w, h))
    cv.imshow("rotated", rotated)

    # draw a 2px thick blue rectangle
    output = resized.copy()
    cv.rectangle(output, (320, 60), (420, 160), (255, 0, 0), 2)
    cv.imshow("rectangle", output)

    # draw text on image
    text_image = resized.copy()
    cv.putText(text_image, "OPENCV IS COOL", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.imshow("text", text_image)

    # show our image
    cv.imshow("my image", img)

    # wait for keypress before continuing code execution
    k = cv.waitKey(0)

    # check which key was pressed
    if k == ord("s"):
        # create a new file called myimage.png
        cv.imwrite("myimage.png", img)

    # exit if escape is pressed
    if k == 27:
        cv.destroyAllWindows()
        break
