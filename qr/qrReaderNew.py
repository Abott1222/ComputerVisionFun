#python3 qrReader.py --image qr.jpg


#alternative approach contours?


import numpy as np
import argparse
import cv2
import math
import collections
#from collections import namedtuple


def distance(cent1, cent2):
    dx = math.fabs(cent1[0] - cent2[0])
    dy = math.fabs(cent1[1] - cent2[1])
    return math.sqrt(dx**2 + dy**2)

#Centroid1 is B
#centroid2 is A
#Corner is outlier we found... C
def lineEquation(centroid1,centroid2,centroidCorner):
    a = -((centroid2[1] - centroid1[1])/(centroid2[0] - centroid1[0]))
    b = 1.0
    c = (((centroid2[1] - centroid1[1])/(centroid2[0] - centroid1[0]))*centroid1[0]) - centroid1[1]
    try:
        pdist = (a*j[0]+(b*centroidCorner[1])+c)/math.sqrt((a**2)+(b**2))
    except:
        return 0
    else:
        return pdist

def lineSlope(cent1, cent2):
    dx = cent2[0] - cent1[0]
    dy = cent2[1] - cent2[1]
    if dy != 0:
        align = 1
        dxy = dy / dx
        return dxy, align
    else:
        align = 0
        dxy = 0.0
        return dxy, align


'''
1  2
3  4
'''
def getVertices(contours, cid, ):
    result = []
    corner1 = (0.0, 0.0)
    corner2 = (0.0, 0.0)
    corner3 = (0.0, 0.0)
    corner4 = (0.0, 0.0)
    x, y, w, h = cv2.boundingRect(contours[cid])






if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                   help="Path to the image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])



    #define new image width to be 150
    #so we need to compute ratio of old height to new height thus..
    r = 450 / image.shape[1]
    dim = (450, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("resized with width: {}".format("450"), resized)
    cv2.waitKey(0)


    image_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)





    (thresh, binary_image) = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    cv2.imshow("thres", binary_image)
    cv2.waitKey(0)



    '''
    #cv2.RETR_LIST -> finds all
    im2, contours_all, hierarchy = cv2.findContours(binary_image.copy(), cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE)
    
    image_copy2 = image.copy()
    cv2.drawContours(image_copy2, contours_all, -1, (0,255,0), 3)
    
    
    cv2.imshow("retr list", image_copy2)
    cv2.waitKey(0)
    '''

    #cv2.RETR_TREE -> Doesnt find anything??
    im2, contours, hierarchy = cv2.findContours(binary_image.copy(), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE)

    image_copy3 = resized.copy()
    cv2.drawContours(image_copy3, contours, -1, (0,0,255), 3)
    cv2.imshow("retr tree!!!!", image_copy3)
    cv2.waitKey(0)


    ### using contours_all as it it includes eveything found by the RETR_LIST retrieval method
    #https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71

    #finding features


    squaresFound = []
    #indexes of squares that were found
    #smallest is always [2] because it was the final child appended to the array
    #[
        #set of contours(index)
        #[1,2,3],

        #next set of contours
        #[10,11,12]
    #]

    centroids = []
    #print("Hierachy is {}\n".format(hierarchy[0]))
    print(contours[0])
    for i in range(len(contours)):
        count = 0
        k = i
        squareIndices = []
        squareIndices.append(i)
        while(hierarchy[0][k][2] != -1):
            k = hierarchy[0][k][2]
            count = count +1
            squareIndices.append(k)
        if(count > 1):
            squaresFound.append(squareIndices)
            print("Squares found array: {}".format(squaresFound))


    image_final = resized.copy()
    contourProperties = {}

    for i in range(len(squaresFound)):

        smallestSquareIndex = squaresFound[i][2]
        M = cv2.moments(contours[smallestSquareIndex])

        centX = M['m10'] / M['m00']
        centY = M['m01'] / M['m00']

        if i not in contourProperties:
            contourProperties[i] = {}
            contourProperties[i]["centroid"] = (centX, centY)

        #draw contours...
        for contourIndex in squaresFound[i]:
            cnt = contours[contourIndex]
            cv2.drawContours(image_final, [cnt], 0, ((25*i)%255, 255, (15*i)%255), 3)

    cv2.imshow("squares we found", image_final)
    cv2.waitKey(0)
    


    A = contourProperties[0]["centroid"]
    B = contourProperties[1]["centroid"]
    C = contourProperties[2]["centroid"]

    cent2Index = {}
    cent2Index[A] = 0
    cent2Index[B] = 1
    cent2Index[C] = 2


    AB = distance(A,B)
    AC = distance(A,C)
    BC = distance(B,C)

    #largest is always hypot which means the point not included is our corner
    if (AB > BC and AB > AC):
        top = C
        median1 = A
        median2 = B
    elif (AC > AB and AC > BC):
        top = B
        median1 = A
        median2 = C
    elif (BC > AB and BC > AC):
        top = A
        median1 = B
        median2 = C


    dist = lineEquation(median1, median2, top)
    slope, align = lineSlope(median1, median2)


    # Centroid1 is B
    # centroid2 is A
    # Corner is outlier we found... C
    #dist = lineEquation(median1, median2, top)
    #def lineEquation(centroid1, centroid2, centroidCorner):

    #checking if both dist and slope pos
    if dist == math.fabs(dist) and slope == math.fabs(dist):
        #A bottom... B right
        print("got here 1")
        bottom = median1
        right = median2
    elif slope != math.fabs(dist) and dist == math.fabs(dist):
        #A right ... B bottom
        print("got here 2")
        bottom = median2
        right = median1
    elif slope == math.fabs(dist) and dist != math.fabs(dist):
        # A right ... B bottom
        print("got here 3")
        bottom = median2
        right = median1
    else:
        # A bottom... B right
        print("got here 4")
        bottom = median1
        right = median2
    print("median1: {} median2: {}".format(median1, median2))

    indexOfTop = cent2Index[top]
    topContArray = squaresFound[indexOfTop]


    indexOfBottom = cent2Index[bottom]
    bottomContArray = squaresFound[indexOfBottom]

    indexOfRight = cent2Index[right]
    rightContArray = squaresFound[indexOfRight]

    testingOrientation = resized.copy()
    testingBoundedBoxCorners = resized.copy()
    count = 0
    for ix in [topContArray[0], bottomContArray[0], rightContArray[0]]:
        cnt = contours[ix]

        x, y, w, h = cv2.boundingRect(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(testingBoundedBoxCorners, (x, y), (x + w, y + h), (150, 255, 35), 2)

        print("contour index is: {}".format(ix))
        print("the set of contours are: {}". format(cnt))

        #remember is is bgr instead of rgb with opencv
        if count == 0:
            color = (255, 0, 0)
        elif count == 1:
            color = (0, 255, 0)
        else:
            color = (0,0,255)
        cv2.drawContours(testingOrientation, [cnt], 0, color, 3)

        count = count +1

    cv2.imshow("testing orientation... top is {} bottom is {} right is {}".format("blue", "green", "red"), testingOrientation)
    cv2.waitKey(0)

    cv2.imshow("testing accuracy of bounded box", testingBoundedBoxCorners)
    cv2.waitKey(0)






    # #### Identify the four corners of each Identification markers #### #
    #corner can be calculated as the 4 points of the contour that is furthest from the centroid
    #get vertices of example code...

    print("\n\n")
    print("len of contours is: {}".format(len(contours[0])))
    print(contours[0][0])
    print("\n\n")

    print(contours[1][0][0][0])

    #for contour in contours:
        #print(contour)
        #print(type(contour))





















