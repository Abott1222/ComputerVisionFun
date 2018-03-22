import numpy as np
import argparse
import cv2
import math

#import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
               help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

cv2.imshow("Origional", image)
cv2.waitKey(0)



image1 = cv2.imread("image.png")
image2 = cv2.imread("dog2.jpg")

#Need to be same size and same number of channels
####### RESIZING ########

dim = (450, 450)
resized1 = cv2.resize(image1, dim, interpolation = cv2.INTER_AREA)
resized2 = cv2.resize(image2, dim, interpolation = cv2.INTER_AREA)





dst = cv2.addWeighted(resized1,0.7,resized2,0.3,0)
dst2 = cv2.add(resized2, resized1)


cv2.imshow("Blended w/o weights", dst2)
cv2.waitKey(0)

cv2.imshow("Blended with weights", dst)
cv2.waitKey(0)




