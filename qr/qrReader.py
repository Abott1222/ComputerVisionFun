#python3 qrReader.py --image qr.jpg


#alternative approach contours?


import numpy as np
import argparse
import cv2
import math
import collections
#from collections import namedtuple





ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
               help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(len(image_gray.shape))
cv2.imshow("grey", image_gray)
cv2.waitKey(0)



#ret, thresh = cv2.threshold(image_grey, 127, 255, 0)

#convert to binary because contours only works with black and white(1 channel images)
(thresh, binary_image) = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


cv2.imshow("thres", binary_image)
cv2.waitKey(0)

###########trying contours with different modes


#cv.RETR_EXTERNAL -> finds extreme(boundary)
im2, contours, hierarchy = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_NONE)
image_copy1 = image.copy()
cv2.drawContours(image_copy1, contours, -1, (255,255,0), 3)


cv2.imshow("contours", image_copy1)
cv2.waitKey(0)


#cv2.RETR_LIST -> finds all
im2, contours_all, hierarchy = cv2.findContours(binary_image.copy(), cv2.RETR_LIST,
	cv2.CHAIN_APPROX_NONE)

image_copy2 = image.copy()
cv2.drawContours(image_copy2, contours_all, -1, (0,255,0), 3)


cv2.imshow("retr list", image_copy2)
cv2.waitKey(0)


#cv2.RETR_TREE -> Doesnt find anything??
im2, contours_all_two, hierarchy = cv2.findContours(binary_image.copy(), cv2.RETR_TREE,
	cv2.CHAIN_APPROX_NONE)

image_copy3 = image.copy()
cv2.drawContours(image_copy2, contours, -1, (0,255,0), 3)


### using contours_all as it it includes eveything found by the RETR_LIST retrieval method
#https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71

#finding features

#print("len of contours: {}".format(len(contours_all)))


contourDict = {}
squaresFound = []
#dataAndIndex = namedtuple("dataAndIndex", ["data", "index"])
#f = Fruit(name="banana", color="red")




print("Hierachy is {}\n".format(hierarchy[0]))
for i in range(len(contours_all_two)):
    count = 0
    k = i
    while(hierarchy[0][k][2] != -1):
        k = hierarchy[0][k][2]
        count = count +1
        print("Got here\n")
        print("k is {} && count is {}".format(k, count))
    if(count > 1):
        squaresFound.append(contours_all[k])


    cnt = contours_all[i]
    area = cv2.contourArea(cnt)
    print("\n\n")
    print("Area is {}".format(area))
    print("Hierachy is {}".format(hierarchy[0][i]))
    print("\n\n")
    M = cv2.moments(cnt)

    #centroid calc x=m10/m00 y= m01/m00
    if(M['m00'] != 0):
        centX = M['m10']/M['m00']
        centY = M['m01'] / M['m00']
        coords = (centX, centY)
        if(centY and centY):
            print("centroid is {},{}".format(centX, centY))
            #tmp = Coord(x=centX, y=centY)
            if coords not in contourDict:
                contourDict[coords] = [1,[i]]
            else:
                tmp = contourDict[coords]
                tmp[0] = tmp[0] + 1
                tmp[1].append(i)

                contourDict[coords] = contourDict[coords]


od = collections.OrderedDict(sorted(contourDict.items()))
print("\n\n")
for key in od.keys():
    print(key)
print("\n\n")


for key in contourDict.keys():
    print("key is: {}".format(key))
    print(contourDict[key])
    if(contourDict[key][0] > 2):
        #squaresFound.append(contourDict[key][1])
        continue

#print("Squares found: {}".format(len(squaresFound)))




cv2.imshow("contours", image_copy3)
cv2.waitKey(0)

image_final = image.copy()
print("squares found: {}".format(squaresFound))
for arrayOfContours in squaresFound:
    for i in arrayOfContours:
        cnt = contours_all[i]
        cv2.drawContours(image_final, [cnt], 0, ((25*i)%255, 255, (15*i)%255), 3)


cv2.imshow("squares found", image_final)
cv2.waitKey(0)

##getting features

