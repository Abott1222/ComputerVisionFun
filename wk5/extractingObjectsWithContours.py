#http://blog.ayoungprogrammer.com/2013/01/equation-ocr-part-1-using-contours-to.html/
#matlab example -- https://www.mathworks.com/help/vision/examples/automatically-detect-and-recognize-text-in-natural-images.html
#python3 extractingObjectsWithContours.py --image letters.jpg


##if we need blank canvas for later..
##canvas = np.zeros((colSize+1,rowSize+1,3), dtype="uint8")

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


#canvas = np.zeros((colSize+1,rowSize+1,3), dtype="uint8")



####### RESIZING ########

#define new image width to be 150
#so we need to compute ratio of old height to new height thus..
r = 450 / image.shape[1]
dim = (450, int(image.shape[0] * r))
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

cv2.imshow("resized with width: {}".format("450"), resized)
cv2.waitKey(0)