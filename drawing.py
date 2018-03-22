import numpy as np
import cv2

canvas = np.zeros((300,300,3), dtype="uint8")

green = (0,255,0)
cv2.line(canvas, (0,0), (300,300), green)
cv2.imshow("canvas", canvas)
cv2.waitKey(0)

red = (0,0,255)
cv2.line(canvas, (300,0), (0,300), red, 3)
cv2.imshow("canvas", canvas)
cv2.waitKey(0)

cv2.rectangle(canvas, (300,0), (0,300), red, 3)
cv2.imshow("canvas", canvas)
cv2.waitKey(0)


#50 x 50
#negative thickness means fill it in
cv2.rectangle(canvas, (10,10), (60,60), green, -1)
cv2.imshow("canvas", canvas)
cv2.waitKey(0)



# Circles
canvas = np.zeros((300,300,3), dtype="uint8")

(centerX, centerY) = (int(canvas.shape[1] /2), int(canvas.shape[0] /2))
white = (255,255,255)

for radius in range(0, 175, 25):
    print("x: {}, y: {}, radius: {}".format(centerX, centerY, radius))
    cv2.circle(canvas, (centerX, centerY), radius, white)
cv2.imshow("canvas", canvas)
cv2.waitKey(0)