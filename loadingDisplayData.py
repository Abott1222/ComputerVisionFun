#python3 loadingDisplayData.py --image image.png

import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
               help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])



print( "Width {} pixels".format(image.shape[1]) )
print( "Height {} pixels".format(image.shape[0]) )
print( "Channels {} ".format(image.shape[2]) )

cv2.imshow("Origional", image)
cv2.waitKey(0)


#openCV stores colors in reverse order
(b,g,r) = image[0,0]
print("pixels at (0,0) - Red {}, Green {}, Blue {}".format(r,g,b))

#pure red
image[0,0] = (0,0,255)

(b,g,r) = image[0,0]

print("pixels at (0,0) - Red {}, Green {}, Blue {}".format(r,g,b))

corner = image[0:100, 0:25]
cv2.imshow("Corner",corner)

image[0:100, 0:25] = (0,0,255)

cv2.imshow("Updated image", image)
cv2.waitKey(0)

#newImage = np.zeros((300,300,3), dtype="uint8")

#cv2.imwrite("newimage.jpg", image)


