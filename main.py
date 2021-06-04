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

# extract a 50 by 50 pixel square ROI (region of interest) from the input image starting at x=40, y=60
roi = img[60:110, 40:90]
cv.imshow("ROI", roi)
# wait for keypress before continuing code execution
cv.waitKey(0)

# resize image
resized = cv.resize(img, (500, 500))
cv.imshow("resized", resized)
cv.waitKey(0)

# show our image
cv.imshow("my image", img)
k = cv.waitKey(0)

# check which key was pressed
if k == ord("s"):
    # create a new file called myimage.png
    cv.imwrite("myimage.png", img)

