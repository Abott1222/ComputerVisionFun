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



## setup plot with matplotlib

plt.figure()
plt.title(" 'flattened' color hist")
plt.xlabel("bins")
plt.ylabel("# pixels per bin")


##### calculating histograms

#cv2.calcHist(images, channels, mask, bins, ranges)



#######  calc COLOR hist STEPS #######
    # 1) split image into color channels not bgr because openCV stores image in numPy in REVERSE order

chans = cv2.split(resized)
    #note not rgb because reversed as mentioned above
colors = ("b", "g", "r")

for (chan, color) in zip(chans, colors):
    #same as grayscale code except we are calculating hist for each channel
    hist = cv2.calcHist([chan], [0], None, [256], [0,256])
    plt.plot(hist, color = color)
    plt.xlim([0,256])


##plot with matplotlib *Note: Setup done at top

#plot histogram
plt.plot(hist)


plt.show()
cv2.waitKey(0)




#### 2D Histo

fig = plt.figure()

#2D color histo for G and B
ax = fig.add_subplot(131)
hist = cv2.calcHist([chans[1], chans[0]], [0,1], None, [32,32], [0,256,0,256])
p = ax.imshow(hist, interpolation= "nearest")
ax.set_title("2D color histo for G and B")
plt.colorbar(p)

#2D color histo for G and B
ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1], chans[2]], [0,1], None, [32,32], [0,256,0,256])
p = ax.imshow(hist, interpolation= "nearest")
ax.set_title("2D color histo for G and R")
plt.colorbar(p)

#2D color histo for G and B
ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0,1], None, [32,32], [0,256,0,256])
p = ax.imshow(hist, interpolation= "nearest")
ax.set_title("2D color histo for B and R")
plt.colorbar(p)

plt.show()
cv2.waitKey(0)








