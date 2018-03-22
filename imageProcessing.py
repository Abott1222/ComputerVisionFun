#python3 loadingDisplayData.py --image image.png
import numpy as np
import argparse
import cv2

import imutils


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
               help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

cv2.imshow("Origional", image)
cv2.waitKey(0)



####### SHIFTING ########

M = np.float32([ [1,0,25], [0,1,50] ])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow("shifted down and right", shifted)
cv2.waitKey(0)


M = np.float32([ [1,0,-50], [0,1,-90] ])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow("shifted up and left", shifted)
cv2.waitKey(0)


#using our utility library
shifted = imutils.translate(image, 0, 100)
cv2.imshow("100px down", shifted)
cv2.waitKey(0)


####### ROTATIONS ########

(h,w) = image.shape[:2]
center = (w/2, h/2)

M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, M, (w,h))
cv2.imshow("rotated by 45 degrees", rotated)
cv2.waitKey(0)

M = cv2.getRotationMatrix2D(center, -90, 1.0)
rotated = cv2.warpAffine(image, M, (w,h))
cv2.imshow("rotated by -90 degrees", rotated)
cv2.waitKey(0)

rotated = imutils.rotate(image,180, scale=0.5)
cv2.imshow("rotated 180 scaled by 1/2", rotated)
cv2.waitKey(0)


####### RESIZING ########

#define new image width to be 150
#so we need to compute ratio of old height to new height thus..
r = 150 / image.shape[1]
dim = (150, int(image.shape[0] * r))
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

cv2.imshow("resized with width: {}".format("150"), resized)
cv2.waitKey(0)