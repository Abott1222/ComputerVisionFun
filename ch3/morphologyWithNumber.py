#python3 temp.py --image ../image.png

#docs -- https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#exercises

#alternative approach contours?


import numpy as np
import argparse
import cv2
import math

#import imutils
colors = [
(60, 180, 75),
(255, 225, 25),
(0, 130, 200),
(245, 130, 48),
(145, 30, 180),
(70, 240, 240),
(240, 50, 230),
(210, 245, 60),
(250, 190, 190),
(0, 128, 128),
(230, 190, 255),
(170, 110, 40),
(255, 250, 200),
(128, 0, 0),
(170, 255, 195)
]



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
               help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

cv2.imshow("Origional", image)
cv2.waitKey(0)


####### RESIZING ########

#define new image width to be 150
#so we need to compute ratio of old height to new height thus..
r = 450 / image.shape[1]
dim = (450, int(image.shape[0] * r))
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

cv2.imshow("resized with width: {}".format("450"), resized)
cv2.waitKey(0)


####convert to binary image
# https://stackoverflow.com/questions/7624765/converting-an-opencv-image-to-black-and-white

# 1 Convert image to greyscale

gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

cv2.imshow("Grey image", gray_image)
cv2.waitKey(0)

# 2 Convert from greyscale to binary image with an unkown threshold

###In this case, the function determines the optimal threshold value using the Otsu’s algorithm and uses it instead of the specified thresh
(thresh, binary_image) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow("Binary image", binary_image)
cv2.waitKey(0)




### opening to reduce noise
##dont need close because there is no noise OUTSIDE of coins...

# Didn't work: (3,3), (4,4), (5,5), (6,6) ... (11,11) :(
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
res = cv2.morphologyEx(binary_image,cv2.MORPH_OPEN,kernel)

cv2.imshow("reduced noise with ellipse", res)
cv2.waitKey(0)



### Counting number of objects with connected components

#4 neighbors
output = cv2.connectedComponentsWithStats(res, 8, cv2.CV_32S)
num_labels = output[0]
labels = output[1]
stats = output[2]
centroids = output[3]

print(num_labels) ##Wrong because foreground needs to be white...



image_inverted = cv2.bitwise_not(res)

cv2.imshow("inverted image", image_inverted)
cv2.waitKey(0)
output = cv2.connectedComponentsWithStats(image_inverted, 8, cv2.CV_32S)
num_labels = output[0]
print(num_labels)



#### OR ANOTHER APPROACH.... Nope still finding 2 extra shapes...

th, im_th = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY_INV);

# Copy the thresholded image.
im_floodfill = im_th.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0, 0), 255);

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv

output = cv2.connectedComponentsWithStats(image_inverted, 8, cv2.CV_32S)
num_labels = output[0]
centroids = output[3]
stats = output[2]

area = 450 * int(image.shape[0] * r)
print("shape of image: {}.....num of components: {} .... centroids: {}...... stats: {}".format(area ,num_labels,centroids, stats))

#### seeing where centroid of each connected component is...


green = (0, 255, 0)
#for coord in centroids:
    #(r,c) = coord
    #r = int(r)
    #c = int(c)
    #print("r is: {}, c is: {}".format(r,c))
    #cv2.rectangle(resized, (r-50,c-50), (r+50, c+50), green, 5 )

for i in range(len(centroids)):
    (r,c) = centroids[i]
    area = stats[i][4]
    print("Area is: {}".format(area))
    r = int(r)
    c = int(c)
    print("r is: {}, c is: {}".format(r,c))

    movement = math.ceil(math.sqrt(area)/2)


    cv2.rectangle(image, (r-movement,c-movement), (r+movement, c+movement), colors[i], 5 )
    cv2.rectangle(resized, (r - movement, c - movement), (r + movement, c + movement), colors[i], 5)

cv2.imshow("finding centroids...", resized)
cv2.waitKey(0)

cv2.imshow("finding centroids...", image)
cv2.waitKey(0)

print("\n\n\n stats: {}".format(stats))


'''
Green:(60, 180, 75)
Yellow :(255, 225, 25)
Blue:(0, 130, 200)
Orange:(245, 130, 48)
Purple:(145, 30, 180)
Cyan:(70, 240, 240)
Magenta:(240, 50, 230)
Lime:(210, 245, 60)
Pink:(250, 190, 190)
Teal:(0, 128, 128)
Lavender:(230, 190, 255)
Brown:(170, 110, 40)
Beige:(255, 250, 200)
Maroon:(128, 0, 0)
Mint		(170, 255, 195)
Olive	(128, 128, 0)
Coral (255, 215, 180)
Navy(0, 0, 128)
Grey	(128, 128, 128)
'''