import argparse
import cv2
import numpy as np

#python3 detectSquares.py --image qr.jpg

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
           help="Path to the image")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]


im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0,255,0), 3)

cv2.imshow("contours", img)
cv2.waitKey(0)

#cnt = contours[4]
#cv2.drawContours(img, [cnt], 0, (0,255,0), 3)