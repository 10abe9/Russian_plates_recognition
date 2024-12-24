import cv2 as cv
import numpy as np


img1 ="C:\\Users\\qdaid\\Downloads\\Telegram Desktop\\01-393.jpg"

image_path = img1
frame = cv.imread(image_path)


# Convert BGR to HSV
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([10,10,10])
upper_blue = np.array([255,255,255])

# Threshold the HSV image to get only blue colors
mask = cv.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv.bitwise_and(frame,frame, mask=mask)

cv.imshow('frame', frame)
cv.imshow('mask', mask)
cv.imshow('res', res)
cv.waitKey(0)
cv.destroyAllWindows()
