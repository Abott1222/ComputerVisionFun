#python3 test.py --image redTest.jpg

#loops through the pixels the right way using C in python
#https://stackoverflow.com/questions/26445153/iterations-through-pixels-in-an-image-are-terribly-slow-with-python-opencv


import argparse
import cv2
import numpy as np


changeInRed = 2

import time







ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
               help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

rowSize = image.shape[1]
colSize = image.shape[0]

canvas = np.zeros((colSize+1,rowSize+1,3), dtype="uint8")

cv2.imshow("origional", image)
cv2.waitKey(0)

cv2.imshow("origional Canvas", canvas)
cv2.waitKey(0)



count = 0
for r in range(image.shape[1]):
    for c in range(image.shape[0]):
        count = count + 1
        #rgb note this is reversed in opencv
        #print("c is {}, r is {}".format(c,r))
        #(b,g,r) = image[c,r]
        redVal = image.item((c,r,2))
        if redVal*changeInRed > 255:
            redVal = 255
        else:
            redVal = redVal * changeInRed

        image.itemset((c,r,2), redVal)

        #print("Old r: {}, b: {}, g: {}".format(r,g,b))
        #print("row is: {}, column is: {}".format(r,c))
        #canvas[c, r] = (255, 255, 255)

        #if red is primary color
        #if max(b,g,r) == r:
            #if(r*changeInRed > 255):
                #r = 255
            #else:
                #r = r*changeInRed
            #print("Changed r: {}, b: {}, g: {}".format(r, g, b))
            #image[c,r] = (b,g,r)

print("count is {}".format(count))
cv2.imshow("red enhanced", image)
cv2.waitKey(0)

#image[:,200:,2] = 0
#cv2.imshow("What does this do??", image)
cv2.waitKey(0)
