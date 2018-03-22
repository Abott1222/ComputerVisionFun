#http://blog.ayoungprogrammer.com/2013/01/equation-ocr-part-1-using-contours-to.html/
#matlab example -- https://www.mathworks.com/help/vision/examples/automatically-detect-and-recognize-text-in-natural-images.html
#python3 workingWithHistograms.py --image letters.jpg


##if we need blank canvas for later..
##canvas = np.zeros((colSize+1,rowSize+1,3), dtype="uint8")



#IDEAS
#Object detection with histograms
#   Faces --

import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt


def draw_image_histogram(image, channels, color='k'):
    hist = cv2.calcHist([image], channels, None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])


def show_grayscale_histogram(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("image", grayscale_image)
    cv2.waitKey(0)
    draw_image_histogram(grayscale_image, [0])
    plt.show()

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



##### calculating histograms

#cv2.calcHist(images, channels, mask, bins, ranges)

## convert to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#calc hist

#gray as 1 channel so [0], No mask, 256 bins, values 0-256
hist = cv2.calcHist([grayscale_image], [0], None, [256], [0,256])

##plot with matplotlib

plt.figure()
plt.title("grayscale hist")
plt.xlabel("bins")
plt.ylabel("# pixels per bin")


#plot histogram
plt.plot(hist)

plt.xlim([0,256])
plt.show()
cv2.waitKey(0)




