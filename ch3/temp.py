#python3 temp.py --image ../image.png

#docs -- https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#exercises


import numpy as np
import argparse
import cv2

#import imutils


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
               help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

cv2.imshow("Origional", image)
cv2.waitKey(0)


####convert to binary image
# https://stackoverflow.com/questions/7624765/converting-an-opencv-image-to-black-and-white

# 1 Convert image to greyscale

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Grey image", gray_image)
cv2.waitKey(0)

# 2 Convert from greyscale to binary image with an unkown threshold

###In this case, the function determines the optimal threshold value using the Otsu’s algorithm and uses it instead of the specified thresh
(thresh, binary_image) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow("Binary image", binary_image)
cv2.waitKey(0)

### closing to reduce noise
structuringElement = np.ones((3,3),np.uint8)

closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE,  structuringElement)
cv2.imshow("reduced noise experiment", closing)
cv2.waitKey(0)



#### Connected Components labeling ####
#http://answers.opencv.org/question/168281/calculating-image-moments-after-connected-component-labeling-function/


#The second parameter is 4-neighborhood?
#connectivity	8 or 4 for 8-way or 4-way connectivity respectively
#ltype	output image label type. Currently CV_32S and CV_16U are supported.

#Why do these two give seperate outputs?
output = cv2.connectedComponentsWithStats(binary_image, connectivity=4, ltype=cv2.CV_32S)
output = cv2.connectedComponentsWithStats(binary_image, 4, cv2.CV_32S)

num_labels = output[0]
labels = output[1]
stats = output[2]
centroids = output[3]

print("Num labels: {}".format(num_labels))
print("labels: {}".format(labels))
print("Stats: {}".format(stats))




#custom example with numpy
# 0 1 0
# 1 0 1
# 0 1 0
#structuringElement3 = np.array([[0,1,0],[1,0,1], [0,1,0]])

structuringElementOnes = np.ones((5,5),np.uint8)


################   erosion  ########################
erosion = cv2.erode(binary_image, structuringElementOnes ,iterations = 1)
cv2.imshow("Eroded image", erosion)
cv2.waitKey(0)

for i in range(2,4,1):
    erosion2 = cv2.erode(binary_image, structuringElementOnes ,iterations = i)
    cv2.imshow("Eroded image in loop at {}".format(i), erosion2)
    cv2.waitKey(0)


################   dilation  ########################
dilation = cv2.dilate(binary_image,structuringElementOnes,iterations = 1)
cv2.imshow("dilated image", dilation)
cv2.waitKey(0)


####### gradient ############
#It is the difference between dilation and erosion of an image.

gradient = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, structuringElementOnes)
cv2.imshow("dif between dilation and erosion", gradient)
cv2.waitKey(0)



##### Same with cv2.createStructuring element

cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
rect = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

SArray = {"cross":cross, "ellipse":ellipse, "rect":rect}




for elem in SArray.keys():
    erosion = cv2.erode(binary_image, SArray[elem] ,iterations = 1)
    cv2.imshow("Eroded image with Struct image: {}".format(elem), erosion)
    cv2.waitKey(0)

    dilation = cv2.dilate(binary_image,SArray[elem],iterations = 1)
    cv2.imshow("dilated image with Struct image: {}".format(elem), dilation)
    cv2.waitKey(0)

    gradient = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, SArray[elem])
    cv2.imshow("dif between dilation and erosion with Struct image: {}".format(elem), gradient)
    cv2.waitKey(0)

