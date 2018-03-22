import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
           help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

rowSize = image.shape[1]
colSize = image.shape[0]

cv2.imshow("origional image", image)
cv2.waitKey(0)


canvas = np.zeros((colSize+1,rowSize+1,3), dtype="uint8")
cv2.imshow("origional canvas", canvas)
cv2.waitKey(0)

redScale = 2

for row in range(image.shape[1]):
    for c in range(image.shape[0]):
        (b,g,r) = image[c,row]

        if r*redScale > 255:
            r = 255
        else:
            r = r * redScale

        canvas[c, row] = (b, g, r)


cv2.imshow("changed canvas", canvas)
cv2.waitKey(0)